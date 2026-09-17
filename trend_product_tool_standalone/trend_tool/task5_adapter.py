from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .config import CrawlQuery
from .crawler import CandidateImage


ProgressLogger = Callable[[str], None]


class Task5ProcessCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Task5TrendConfig:
    niche: str
    output_dir: Path
    region: str = "US"
    trend_type: str = "growing"
    interest: str = ""
    keyword_limit: int = 50
    max_trends: int = 20
    min_semantic_fit: float = 35.0
    gemini_backend: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    token_path: Path | None = None
    cancel_event: Event | None = None


@dataclass(frozen=True)
class Task5CrawlerConfig:
    package_path: Path
    output_dir: Path
    provider: str = "pinterest-browser"
    max_images_per_query: int = 30
    max_trends: int = 5
    max_queries_per_trend: int = 6
    max_downloads: int = 250
    top_images: int = 100
    vision_mode: str = "auto"
    crawl_purpose: str = "product"
    product_focus: str = "auto"
    gemini_backend: str = "auto"
    vision_model: str = "gemini-2.5-flash"
    refresh_vision_cache: bool = True
    cancel_event: Event | None = None


def run_task5_trend_finder(config: Task5TrendConfig, progress: ProgressLogger | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "pinterest" / "pinterest_trend_finder.py"
    if not script.exists():
        raise RuntimeError(f"Bundled Pinterest trend finder not found: {script}")

    token_path = resolve_task5_token_path(config.token_path, repo_root)
    if config.token_path and not token_path:
        raise RuntimeError(f"Pinterest OAuth token file not found: {config.token_path}")
    if not token_path and not has_task5_env_auth():
        raise RuntimeError(
            "Pinterest OAuth token not found. Provide pinterest/.pinterest_oauth_tokens.json, "
            "PINTEREST_ACCESS_TOKEN, or app credentials for trend discovery."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--niche",
        config.niche,
        "--region",
        config.region,
        "--trend-type",
        config.trend_type,
        "--keyword-limit",
        str(config.keyword_limit),
        "--output",
        str(config.output_dir),
        "--max-trends",
        str(config.max_trends),
        "--min-semantic-fit",
        str(config.min_semantic_fit),
        "--gemini-backend",
        config.gemini_backend,
        "--gemini-model",
        config.gemini_model,
    ]
    if token_path:
        command.extend(["--token-path", str(token_path)])
    if config.interest:
        command.extend(["--interest", config.interest])

    log_path = config.output_dir / "task5_trend_finder.log"
    completed = run_process(command, repo_root, log_path, progress=progress, cancel_event=config.cancel_event)
    if completed.returncode != 0:
        raise RuntimeError(build_process_error("Pinterest trend discovery", completed, log_path))

    package_path = config.output_dir / "trend_package.json"
    if not package_path.exists():
        raise RuntimeError(f"Pinterest trend discovery did not create {package_path}")
    return package_path


def run_task5_image_crawler(config: Task5CrawlerConfig, progress: ProgressLogger | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "pinterest" / "hot_image_crawler.py"
    if not script.exists():
        raise RuntimeError(f"Bundled Pinterest image crawler not found: {script}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    hot_images_path = config.output_dir / "hot_product_images.json"
    if (
        config.max_downloads <= 0
        or config.top_images <= 0
        or config.max_images_per_query <= 0
        or config.max_trends <= 0
    ):
        if progress:
            progress("Crawler: target image download count is 0; skipping Pinterest image crawler.")
        hot_images_path.write_text("[]", encoding="utf-8")
        (config.output_dir / "task5_image_crawler.log").write_text(
            "Pinterest image crawler skipped because one or more crawl limits were set to 0.\n",
            encoding="utf-8",
        )
        return hot_images_path

    command = [
        sys.executable,
        str(script),
        "--input",
        str(config.package_path),
        "--output",
        str(config.output_dir),
        "--provider",
        config.provider,
        "--max-images-per-query",
        str(config.max_images_per_query),
        "--max-trends",
        str(config.max_trends),
        "--max-queries-per-trend",
        str(config.max_queries_per_trend),
        "--max-downloads",
        str(config.max_downloads),
        "--top-images",
        str(config.top_images),
        "--vision-mode",
        config.vision_mode,
        "--crawl-purpose",
        config.crawl_purpose,
        "--product-focus",
        config.product_focus,
        "--gemini-backend",
        config.gemini_backend,
        "--vision-model",
        config.vision_model,
    ]
    if config.refresh_vision_cache:
        command.append("--refresh-vision-cache")

    log_path = config.output_dir / "task5_image_crawler.log"
    completed = run_process(command, repo_root, log_path, progress=progress, cancel_event=config.cancel_event)
    if completed.returncode != 0:
        raise RuntimeError(build_process_error("Pinterest image crawler", completed, log_path))

    if not hot_images_path.exists():
        raise RuntimeError(f"Pinterest image crawler did not create {hot_images_path}")
    return hot_images_path


def has_task5_env_auth() -> bool:
    has_access_token = bool(os.environ.get("PINTEREST_ACCESS_TOKEN", "").strip())
    has_client_credentials = all(
        os.environ.get(name, "").strip()
        for name in ("PINTEREST_APP_ID", "PINTEREST_APP_SECRET", "PINTEREST_SCOPES")
    )
    return has_access_token or has_client_credentials


def task5_default_token_paths(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / ".pinterest_oauth_tokens.json",
        repo_root / "pinterest" / ".pinterest_oauth_tokens.json",
    )


def resolve_task5_token_path(token_path: Path | None, repo_root: Path) -> Path | None:
    if token_path:
        candidate = token_path if token_path.is_absolute() else repo_root / token_path
        return candidate if candidate.exists() else None
    for candidate in task5_default_token_paths(repo_root):
        if candidate.exists():
            return candidate
    return None


def is_task5_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "client_credentials token request failed",
            "oauth token not found",
            "oauth token file not found",
            "has no access_token",
            "pinterest api error 401",
            "pinterest api error 403",
            "not authorized",
            "sufficient permissions",
            "restricted feature",
            "does not have access",
            "winerror 10013",
            "forbidden by its access permissions",
            "failed to establish a new connection",
            "windows blocked the https socket",
            "local network/security rule",
            "max retries exceeded with url: /v5/oauth/token",
            "no trend candidates were collected",
        )
    )


def queries_from_trend_package(package_path: Path, max_queries_per_trend: int = 6) -> tuple[CrawlQuery, ...]:
    payload = json.loads(package_path.read_text(encoding="utf-8-sig"))
    queries: list[CrawlQuery] = []
    seen: set[str] = set()
    for trend in payload.get("trends", []):
        raw_queries = trend.get("queries", [])
        normalized = []
        for index, item in enumerate(raw_queries, start=1):
            if isinstance(item, str):
                normalized.append((index, item))
            elif isinstance(item, dict):
                normalized.append((int(item.get("priority") or index), str(item.get("query") or "")))
        for _, keyword in sorted(normalized)[:max_queries_per_trend]:
            keyword = keyword.strip()
            key = keyword.lower()
            if not keyword or key in seen:
                continue
            seen.add(key)
            queries.append(CrawlQuery(keyword=keyword))
    return tuple(queries)


def candidates_from_hot_product_images(hot_images_path: Path) -> list[CandidateImage]:
    payload = json.loads(hot_images_path.read_text(encoding="utf-8-sig"))
    candidates: list[CandidateImage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        local_path = str(item.get("local_path") or "").strip()
        if not local_path:
            continue
        path = Path(local_path)
        if not path.is_absolute():
            # The bundled crawler is launched from the standalone tool root.
            # Resolve its relative output paths there, never in the old shared
            # workspace that happened to contain this project during development.
            repo_path = Path(__file__).resolve().parents[1] / path
            output_path = hot_images_path.parent / path
            path = repo_path.resolve() if repo_path.exists() else output_path.resolve()
        if not path.exists():
            continue
        candidates.append(
            CandidateImage(
                path=path,
                source=str(item.get("pin_url") or item.get("image_url") or hot_images_path),
                keyword=str(item.get("query") or item.get("trend") or "task5_hot_product"),
                source_role=str(item.get("source_role") or "unknown"),
                metadata=item,
            )
        )
    return candidates


def build_process_error(name: str, completed: subprocess.CompletedProcess[str], log_path: Path) -> str:
    details = tail_text(completed.stderr) or tail_text(completed.stdout) or "No process output captured."
    return f"{name} failed with exit code {completed.returncode}. {details} See {log_path}"


def tail_text(value: str, max_lines: int = 6) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " ".join(lines[-max_lines:])


def write_process_log(path: Path, command: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "COMMAND:",
                " ".join(command),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )


def run_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    *,
    progress: ProgressLogger | None = None,
    cancel_event: Event | None = None,
) -> subprocess.CompletedProcess[str]:
    if cancel_event is not None:
        return run_process_cancellable(command, cwd, log_path, cancel_event, progress)
    if progress:
        return run_process_streaming(command, cwd, log_path, progress)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    write_process_log(log_path, command, completed)
    return completed


def run_process_cancellable(
    command: list[str],
    cwd: Path,
    log_path: Path,
    cancel_event: Event,
    progress: ProgressLogger | None,
) -> subprocess.CompletedProcess[str]:
    if cancel_event.is_set():
        raise Task5ProcessCancelled("Pinterest subprocess cancelled before start.")
    if progress:
        progress(f"Starting: {' '.join(command[:2])}")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    while process.poll() is None:
        if cancel_event.is_set():
            terminate_process_tree(process)
            stdout, stderr = process.communicate()
            completed = subprocess.CompletedProcess(command, process.returncode or -1, stdout=stdout, stderr=stderr)
            write_process_log(log_path, command, completed)
            raise Task5ProcessCancelled("Pinterest subprocess cancelled by user.")
        time.sleep(0.2)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)
    write_process_log(log_path, command, completed)
    return completed


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            text=True,
            capture_output=True,
            check=False,
        )
        return
    process.terminate()


def run_process_streaming(
    command: list[str],
    cwd: Path,
    log_path: Path,
    progress: ProgressLogger,
) -> subprocess.CompletedProcess[str]:
    stdout_lines: list[str] = []
    stderr = ""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        stdout_lines.append(raw_line)
        if line:
            progress(f"Crawler: {line}")
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    completed = subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)
    write_process_log(log_path, command, completed)
    return completed




