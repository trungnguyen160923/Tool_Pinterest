from __future__ import annotations

import csv
import hashlib
import html
import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import sys
from urllib.parse import urlparse


LOG = logging.getLogger("pinterest")


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


load_dotenv(
    [
        project_root() / ".env",
    ]
)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: Any, fallback: str = "item") -> str:
    text = normalize_text(value).replace(" ", "-")
    return text[:80] or fallback


def stable_id(*parts: Any, length: int = 16) -> str:
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return stable_id(raw, length=32)


def unique(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item or "").strip()
        key = normalize_text(text)
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def clamp(value: Any, lo: float = 0.0, hi: float = 100.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(lo, min(hi, number)), 2)


def clamp01(value: Any, default: float = 0.0) -> float:
    return clamp(value, 0.0, 1.0, default)


def valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def truncate_text(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataclass_to_dict(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def html_page(title: str, body: str, extra_css: str = "") -> str:
    safe_title = html.escape(title)
    css = """
    :root { color-scheme: light; font-family: Inter, Arial, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #17202a; }
    header { padding: 28px 34px; background: #ffffff; border-bottom: 1px solid #d8dee8; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    main { padding: 24px 34px 40px; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5e9f0; text-align: left; vertical-align: top; }
    th { font-size: 12px; text-transform: uppercase; color: #526173; background: #f9fafc; }
    .muted { color: #667386; }
    .pill { display: inline-block; padding: 3px 7px; border-radius: 6px; background: #eef2f7; margin: 2px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
    .card { background: #ffffff; border: 1px solid #dfe5ee; border-radius: 8px; overflow: hidden; }
    .card img { width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block; background: #e9edf3; }
    .card .pad { padding: 12px; }
    .score { font-weight: 700; }
    """ + extra_css
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
