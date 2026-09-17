from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from ..shared import utils


DEFAULT_TOKEN_PATHS = [
    utils.project_root() / ".pinterest_oauth_tokens.json",
]


class PinterestApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def explain_network_error(exc: BaseException) -> str:
    text = str(exc)
    if "WinError 10013" in text:
        return (
            "Windows blocked the HTTPS socket to api.pinterest.com:443 "
            "(WinError 10013). This is a local network/security rule, not a "
            "Pinterest token permission error. Check Windows Defender Firewall, "
            "antivirus web shield, VPN/proxy policy, or whether python.exe is "
            "blocked from outbound HTTPS."
        )
    return (
        "Could not connect to Pinterest API. Check internet access, DNS, proxy, "
        "VPN, firewall, and whether api.pinterest.com:443 is reachable."
    )


class PinterestClient:
    def __init__(
        self,
        *,
        token_path: Path | None = None,
        timeout: int = 30,
        app_id: str | None = None,
        app_secret: str | None = None,
    ):
        self.base_url = "https://api.pinterest.com/v5"
        self.timeout = timeout
        self.app_id = app_id or utils.env("PINTEREST_APP_ID")
        self.app_secret = app_secret or utils.env("PINTEREST_APP_SECRET")
        self.token_path = token_path or self._find_token_path()
        self.persist_tokens = self.token_path.exists()
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.8,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.tokens = self._load_tokens()

    @staticmethod
    def _find_token_path() -> Path:
        for path in DEFAULT_TOKEN_PATHS:
            if path.exists():
                return path
        return DEFAULT_TOKEN_PATHS[0]

    def _load_tokens(self) -> dict[str, Any]:
        if self.token_path.exists():
            raw = utils.read_json(self.token_path)
            if not isinstance(raw, dict) or not raw.get("access_token"):
                raise PinterestApiError(f"Pinterest token file has no access_token: {self.token_path}")
            return raw

        env_tokens = self._tokens_from_env()
        if env_tokens:
            return env_tokens

        generated_tokens = self._client_credentials_tokens()
        if generated_tokens:
            return generated_tokens

        raise PinterestApiError(
            "Pinterest OAuth token not found. Provide .pinterest_oauth_tokens.json, "
            "PINTEREST_ACCESS_TOKEN, or PINTEREST_APP_ID/PINTEREST_APP_SECRET/PINTEREST_SCOPES for client_credentials."
        )

    def _tokens_from_env(self) -> dict[str, Any] | None:
        access_token = utils.env("PINTEREST_ACCESS_TOKEN").strip()
        if not access_token:
            return None
        tokens: dict[str, Any] = {
            "access_token": access_token,
            "token_type": utils.env("PINTEREST_TOKEN_TYPE", "bearer") or "bearer",
        }
        optional_fields = {
            "refresh_token": "PINTEREST_REFRESH_TOKEN",
            "scope": "PINTEREST_SCOPES",
            "access_token_expires_at": "PINTEREST_ACCESS_TOKEN_EXPIRES_AT",
            "refresh_token_expires_at": "PINTEREST_REFRESH_TOKEN_EXPIRES_AT",
            "issued_at": "PINTEREST_TOKEN_ISSUED_AT",
        }
        for field, env_name in optional_fields.items():
            value = utils.env(env_name).strip()
            if value:
                tokens[field] = value
        return tokens

    def _client_credentials_tokens(self) -> dict[str, Any] | None:
        scopes = utils.env("PINTEREST_SCOPES").strip()
        if not self.app_id or not self.app_secret or not scopes:
            return None

        auth = base64.b64encode(f"{self.app_id}:{self.app_secret}".encode("utf-8")).decode("ascii")
        try:
            response = self.session.post(
                f"{self.base_url}/oauth/token",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={
                    "grant_type": "client_credentials",
                    "scope": scopes,
                },
                timeout=self.timeout,
            )
        except RequestException as exc:
            raise PinterestApiError(f"{explain_network_error(exc)} Original error: {exc}") from exc

        try:
            payload = response.json()
        except Exception:
            payload = response.text
        if not 200 <= response.status_code < 300:
            raise PinterestApiError(
                "Pinterest client_credentials token request failed. Check app approval, 2FA, client secret, and scopes.",
                response.status_code,
                payload,
            )
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise PinterestApiError("Pinterest client_credentials response had no access_token.", response.status_code, payload)

        now = time.time()
        if payload.get("expires_in") and not payload.get("access_token_expires_at"):
            payload["access_token_expires_at"] = now + float(payload["expires_in"])
        payload.setdefault("issued_at", now)
        payload.setdefault("scope", scopes)
        return payload

    def _save_tokens(self) -> None:
        if self.persist_tokens:
            utils.write_json(self.token_path, self.tokens)

    def _access_token_expired(self) -> bool:
        expires_at = self.tokens.get("access_token_expires_at")
        try:
            return float(expires_at) <= time.time() + 90
        except (TypeError, ValueError):
            return False

    def refresh_access_token_if_needed(self) -> None:
        if not self._access_token_expired():
            return
        refresh_token = self.tokens.get("refresh_token")
        if not refresh_token:
            return
        if not self.app_id or not self.app_secret:
            return

        auth = base64.b64encode(f"{self.app_id}:{self.app_secret}".encode("utf-8")).decode("ascii")
        try:
            response = self.session.post(
                f"{self.base_url}/oauth/token",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=self.timeout,
            )
        except RequestException:
            return
        if not 200 <= response.status_code < 300:
            return
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("access_token"):
            return
        now = time.time()
        self.tokens.update(payload)
        if payload.get("expires_in"):
            self.tokens["access_token_expires_at"] = now + float(payload["expires_in"])
        self.tokens["issued_at"] = now
        self._save_tokens()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.refresh_access_token_if_needed()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        try:
            response = self.session.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.tokens['access_token']}",
                    "Accept": "application/json",
                },
                params=params or {},
                timeout=self.timeout,
            )
        except RequestsConnectionError as exc:
            raise PinterestApiError(f"{explain_network_error(exc)} Original error: {exc}") from exc
        except RequestException as exc:
            raise PinterestApiError(f"{explain_network_error(exc)} Original error: {exc}") from exc
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        if not 200 <= response.status_code < 300:
            raise PinterestApiError(
                f"Pinterest API error {response.status_code} for {path}",
                response.status_code,
                payload,
            )
        return payload
