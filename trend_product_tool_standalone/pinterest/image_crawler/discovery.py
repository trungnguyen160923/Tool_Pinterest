from __future__ import annotations

import html
import atexit
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..shared.models import SearchResult, TrendPackage, TrendPackageItem
from ..shared.utils import env, stable_id, valid_http_url
from ..trend_finder.pinterest_client import PinterestClient


LOG = logging.getLogger("pinterest.discovery")


def bool_env(name: str, default: bool = False) -> bool:
    value = env(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def browser_profile_dir(value: str = "") -> Path:
    raw = value or env("PINTEREST_BROWSER_PROFILE_DIR", "")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / ".pinterest_browser_profile").resolve()


def largest_image_from_payload(obj: Any) -> tuple[str, int | None, int | None]:
    found: list[tuple[int, str, int | None, int | None]] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            url = value.get("url") or value.get("src")
            if isinstance(url, str) and valid_http_url(url):
                width = value.get("width")
                height = value.get("height")
                try:
                    w = int(width) if width is not None else None
                except Exception:
                    w = None
                try:
                    h = int(height) if height is not None else None
                except Exception:
                    h = None
                area = (w or 0) * (h or 0)
                bonus = 0
                lower = path.lower()
                if "original" in lower or "1200" in lower:
                    bonus = 10**12
                elif "736" in lower or "600" in lower:
                    bonus = 10**10
                found.append((bonus + area, url, w, h))
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(obj)
    if not found:
        return "", None, None
    found.sort(key=lambda item: item[0], reverse=True)
    _, url, width, height = found[0]
    return url, width, height


class DiscoveryProvider(ABC):
    name = "base"

    @abstractmethod
    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        ...


class PinterestPartnerPinProvider(DiscoveryProvider):
    name = "pinterest_partner_pin_search"

    def __init__(self, client: PinterestClient):
        self.client = client

    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        payload = self.client.get(
            "/search/partner/pins",
            params={
                "term": query,
                "country_code": region,
                "locale": locale,
                "limit": max(1, min(50, int(limit))),
            },
        )
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            pin_id = str(item.get("id") or "").strip()
            image_url, width, height = largest_image_from_payload(item.get("media") or item)
            if not image_url:
                continue
            results.append(
                SearchResult(
                    result_id=stable_id(self.name, trend.trend_id, query, pin_id, image_url),
                    query=query,
                    trend_id=trend.trend_id,
                    trend=trend.trend,
                    image_url=image_url,
                    pin_url=f"https://www.pinterest.com/pin/{pin_id}/" if pin_id else "",
                    pin_id=pin_id,
                    title=str(item.get("title") or item.get("alt_text") or ""),
                    description=str(item.get("description") or ""),
                    source=self.name,
                    width=width,
                    height=height,
                    raw={"index": index, "id": pin_id},
                )
            )
        return results


class PinterestPinSearchProvider(DiscoveryProvider):
    name = "pinterest_pin_search"

    def __init__(self, client: PinterestClient):
        self.client = client

    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        payload = self.client.get(
            "/search/pins",
            params={
                "query": query,
                "country_code": region,
                "locale": locale,
                "limit": max(1, min(50, int(limit))),
            },
        )
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            pin_id = str(item.get("id") or "").strip()
            image_url, width, height = largest_image_from_payload(item.get("media") or item)
            if not image_url:
                continue
            results.append(
                SearchResult(
                    result_id=stable_id(self.name, trend.trend_id, query, pin_id, image_url),
                    query=query,
                    trend_id=trend.trend_id,
                    trend=trend.trend,
                    image_url=image_url,
                    pin_url=f"https://www.pinterest.com/pin/{pin_id}/" if pin_id else "",
                    pin_id=pin_id,
                    title=str(item.get("title") or item.get("alt_text") or ""),
                    description=str(item.get("description") or ""),
                    source=self.name,
                    width=width,
                    height=height,
                    raw={"index": index, "id": pin_id},
                )
            )
        return results


def normalize_pinimg_url(url: str) -> str:
    url = url.replace("\\/", "/")
    url = html.unescape(url)
    url = re.sub(r"\?.*$", "", url)
    url = re.sub(r"/(?:60x60|75x75|136x136|170x|236x|474x|564x)/", "/736x/", url)
    return url


class PinterestBrowserProvider(DiscoveryProvider):
    name = "pinterest_browser"

    def __init__(
        self,
        timeout: int = 30,
        scrolls: int = 8,
        *,
        user_data_dir: str = "",
        headless: bool | None = None,
    ):
        self.timeout = timeout
        self.scrolls = max(1, int(scrolls))
        self.user_data_dir = browser_profile_dir(user_data_dir)
        self.headless = bool_env("PINTEREST_BROWSER_HEADLESS", False) if headless is None else headless
        self._playwright = None
        self._context = None
        atexit.register(self.close)

    def _cleanup_stale_profile_locks(self) -> None:
        try:
            for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
                lock_file = self.user_data_dir / name
                if lock_file.exists():
                    try:
                        lock_file.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
        if os.name == "nt":
            try:
                import subprocess
                target_dir = str(self.user_data_dir).replace("'", "''")
                ps_cmd = (
                    f"Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                    f"Where-Object {{ $_.CommandLine -like '*{target_dir}*' }} | "
                    f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, timeout=5)
            except Exception:
                pass

    def _ensure_context(self):
        if self._context is not None:
            return self._context
        from playwright.sync_api import sync_playwright

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                if self._playwright is None:
                    self._playwright = sync_playwright().start()
                self._context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.user_data_dir),
                    headless=self.headless,
                    viewport={"width": 1366, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"
                    ),
                    locale=env("PINTEREST_LOCALE", "en-US"),
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--disable-quic",
                        "--disable-http3",
                    ],
                )
                return self._context
            except Exception as exc:
                LOG.warning("Failed to launch Pinterest browser context (attempt %d/2): %s", attempt + 1, exc)
                self.close()
                if attempt == 0:
                    self._cleanup_stale_profile_locks()
                else:
                    raise
        return self._context

    def close(self) -> None:
        for attr in ("_context", "_playwright"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                obj.close() if attr != "_playwright" else obj.stop()
            except Exception:
                pass
            setattr(self, attr, None)

    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        try:
            context = self._ensure_context()
            page = context.new_page()
        except Exception:
            self.close()
            raise
        page.set_default_timeout(max(10_000, self.timeout * 1000))
        try:
            url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
            page.goto(url, wait_until="domcontentloaded", timeout=max(20_000, self.timeout * 1000))
            page.wait_for_timeout(3500)
            if re.search(r"/login|/signup", page.url, re.I):
                raise RuntimeError(
                    f"Pinterest browser profile is signed out. Run browser login first: {self.user_data_dir}"
                )
            for _ in range(self.scrolls):
                page.mouse.wheel(0, 1800)
                page.wait_for_timeout(1200)

            raw_items = page.evaluate(
                """
                () => {
                  const pinAnchors = Array.from(document.querySelectorAll('a[href*="/pin/"]'));
                  const items = pinAnchors.map((anchor) => {
                    const img = anchor.querySelector('img') || anchor.closest('[data-test-id], div')?.querySelector('img');
                    if (!img) return null;
                    const rect = img.getBoundingClientRect();
                    const titleNode = anchor.closest('[data-test-id], div')?.querySelector('[title], h1, h2, h3, span');
                    const title = titleNode ? (titleNode.getAttribute('title') || titleNode.textContent || '') : '';
                    return {
                      src: img.currentSrc || img.src || '',
                      alt: img.alt || '',
                      title,
                      width: img.naturalWidth || Math.round(rect.width) || 0,
                      height: img.naturalHeight || Math.round(rect.height) || 0,
                      href: anchor.href || ''
                    };
                  }).filter(Boolean);
                  if (items.length) return items;
                  return Array.from(document.querySelectorAll('img')).map((img) => {
                    const anchor = img.closest('a');
                    const rect = img.getBoundingClientRect();
                    return {
                      src: img.currentSrc || img.src || '',
                      alt: img.alt || '',
                      title: '',
                      width: img.naturalWidth || Math.round(rect.width) || 0,
                      height: img.naturalHeight || Math.round(rect.height) || 0,
                      href: anchor ? anchor.href : ''
                    };
                  });
                }
                """
            )
            signed_out = page.evaluate(
                """
                () => /login|signup/i.test(location.pathname) ||
                  Array.from(document.querySelectorAll('button, a')).some((el) =>
                    /log in|sign up/i.test(el.textContent || '')
                  )
                """
            )
            if signed_out and not raw_items:
                raise RuntimeError(
                    f"Pinterest browser profile appears signed out. Run browser login first: {self.user_data_dir}"
                )
        finally:
            page.close()

        output: list[SearchResult] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_items if isinstance(raw_items, list) else []):
            if not isinstance(item, dict):
                continue
            image_url = normalize_pinimg_url(str(item.get("src") or ""))
            if "i.pinimg.com" not in image_url or not re.search(r"\.(jpg|jpeg|png|webp)$", image_url, re.I):
                continue
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            if width < 150 or height < 150:
                continue
            if image_url in seen:
                continue
            seen.add(image_url)
            pin_url = str(item.get("href") or "")
            pin_id_match = re.search(r"/pin/(\d+)", pin_url)
            pin_id = pin_id_match.group(1) if pin_id_match else ""
            output.append(
                SearchResult(
                    result_id=stable_id(self.name, trend.trend_id, query, image_url),
                    query=query,
                    trend_id=trend.trend_id,
                    trend=trend.trend,
                    image_url=image_url,
                    pin_url=pin_url if "/pin/" in pin_url else "",
                    pin_id=pin_id,
                    title=str(item.get("title") or item.get("alt") or ""),
                    source=self.name,
                    width=width,
                    height=height,
                    raw={
                        "search_url": url,
                        "index": index,
                        "profile_dir": str(self.user_data_dir),
                        "headless": self.headless,
                    },
                )
            )
            if len(output) >= limit:
                break
        return output


class PinterestSearchPageProvider(DiscoveryProvider):
    name = "pinterest_search_page"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.8,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 PinterestHotTrendCrawler/1.0",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
        response = self.session.get(url, timeout=self.timeout)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Pinterest search page HTTP {response.status_code}")
        text = response.text
        urls = []
        pattern = r'https?:\\?/\\?/i\.pinimg\.com/[^"\\\s)]+?\.(?:jpg|jpeg|png|webp)(?:\?[^"\\\s)]*)?'
        for match in re.finditer(pattern, text, flags=re.I):
            image_url = normalize_pinimg_url(match.group(0))
            if valid_http_url(image_url):
                urls.append(image_url)
        output: list[SearchResult] = []
        seen: set[str] = set()
        for index, image_url in enumerate(urls):
            if image_url in seen:
                continue
            seen.add(image_url)
            output.append(
                SearchResult(
                    result_id=stable_id(self.name, trend.trend_id, query, image_url),
                    query=query,
                    trend_id=trend.trend_id,
                    trend=trend.trend,
                    image_url=image_url,
                    source=self.name,
                    raw={"search_url": url, "index": index},
                )
            )
            if len(output) >= limit:
                break
        return output


class AutoDiscoveryProvider(DiscoveryProvider):
    name = "auto"

    def __init__(self, providers: list[DiscoveryProvider]):
        self.providers = providers
        self.disabled: set[str] = set()

    @staticmethod
    def is_permission_error(exc: Exception) -> bool:
        text = str(exc).lower()
        payload = getattr(exc, "payload", "")
        text = f"{text} {payload}".lower()
        return any(
            marker in text
            for marker in (
                "401",
                "sufficient permissions",
                "restricted feature",
                "does not have access",
                "missing:",
                "not authorized",
                "signed out",
                "run browser login first",
            )
        )

    def search(
        self,
        *,
        query: str,
        trend: TrendPackageItem,
        limit: int,
        region: str,
        locale: str,
    ) -> list[SearchResult]:
        errors = []
        for provider in self.providers:
            if provider.name in self.disabled:
                continue
            try:
                results = provider.search(
                    query=query,
                    trend=trend,
                    limit=limit,
                    region=region,
                    locale=locale,
                )
                if results:
                    return results
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                if self.is_permission_error(exc):
                    self.disabled.add(provider.name)
                    LOG.warning(
                        "Discovery provider disabled for this run (%s): %s",
                        provider.name,
                        exc,
                    )
                else:
                    LOG.warning("Discovery provider failed for query %r: %s", query, exc)
        if errors:
            raise RuntimeError("; ".join(errors))
        return []


def provider_from_name(name: str, *, timeout: int = 30, token_path: str = "") -> DiscoveryProvider:
    name = name.lower()
    if name in {"auto", "pinterest-browser"}:
        return PinterestBrowserProvider(timeout=timeout)
    if name == "pinterest-web":
        raise ValueError("pinterest-web provider is disabled; use pinterest-browser.")
    if name == "pinterest-api":
        raise ValueError("pinterest-api provider is disabled; use pinterest-browser.")
    if name == "bing-images":
        raise ValueError("bing-images provider is disabled; use pinterest-browser.")
    raise ValueError(f"Unknown discovery provider: {name}")


def load_trend_package(path: str) -> TrendPackage:
    from ..shared.models import QuerySpec, TrendPackage, TrendPackageItem
    from ..shared.utils import read_json

    raw = read_json(__import__("pathlib").Path(path))
    trends: list[TrendPackageItem] = []
    for item in raw.get("trends") or []:
        if not isinstance(item, dict):
            continue
        queries = []
        for query in item.get("queries") or []:
            if isinstance(query, str):
                queries.append(QuerySpec(query=query))
            elif isinstance(query, dict):
                queries.append(
                    QuerySpec(
                        query=str(query.get("query") or ""),
                        intent=str(query.get("intent") or "product"),
                        priority=int(query.get("priority") or len(queries) + 1),
                    )
                )
        trends.append(
            TrendPackageItem(
                trend_id=str(item.get("trend_id") or ""),
                trend=str(item.get("trend") or ""),
                trend_strength=float(item.get("trend_strength") or 0),
                relationship=str(item.get("relationship") or ""),
                semantic_fit=float(item.get("semantic_fit") or 0),
                queries=[query for query in queries if query.query],
                reason=str(item.get("reason") or ""),
                sources=list(item.get("sources") or []),
                source_metrics=dict(item.get("source_metrics") or {}),
                tags=list(item.get("tags") or []),
            )
        )
    return TrendPackage(
        schema_version=str(raw.get("schema_version") or "1.0"),
        generated_at=str(raw.get("generated_at") or ""),
        niche=str(raw.get("niche") or ""),
        region=str(raw.get("region") or "US"),
        source=dict(raw.get("source") or {}),
        trends=trends,
        rejected_trends=list(raw.get("rejected_trends") or []),
        raw_summary=dict(raw.get("raw_summary") or {}),
    )
