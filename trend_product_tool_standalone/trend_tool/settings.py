from __future__ import annotations

import os
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
# The API package is self-contained. Do not load configuration from its former
# workspace or any sibling experiment directory.
REPO_ROOT = TOOL_ROOT


def load_tool_env(*, override: bool = False) -> None:
    for path in (TOOL_ROOT / ".env", REPO_ROOT / ".env"):
        load_env_file(path, override=override)


def load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def env(name: str, default: str = "") -> str:
    load_tool_env()
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = env(name, "")
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    value = env(name, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def env_path(*names: str) -> Path | None:
    for name in names:
        value = env(name, "").strip()
        if value:
            return Path(value)
    return None


def task5_token_path_from_env() -> Path | None:
    configured = env_path("PINTEREST_OAUTH_TOKEN_PATH", "PINTEREST_TOKEN_PATH", "TASK5_TOKEN_PATH")
    if configured:
        return configured
    for candidate in (
        TOOL_ROOT / ".pinterest_oauth_tokens.json",
        TOOL_ROOT / "pinterest" / ".pinterest_oauth_tokens.json",
    ):
        if candidate.exists():
            return candidate
    return None



