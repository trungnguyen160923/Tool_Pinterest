from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "web_ui"
STATUS_LOCK = threading.Lock()
STATUS = {
    "running": False,
    "job": "",
    "returncode": None,
    "log": [],
}


def load_local_env() -> None:
    for path in (ROOT / ".env", ROOT.parent / "task5_craw_hottrend_img" / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


def python_executable() -> str:
    configured = os.environ.get("HOT_TREND_PYTHON") or os.environ.get("PYTHON_EXE")
    if configured:
        return configured

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        venv_python = Path(virtual_env) / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)

    local_venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if local_venv.exists():
        return str(local_venv)

    return sys.executable


def safe_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("Path is outside task5_hottrend.")
    return candidate


def read_json(path: Path, fallback):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}
    return fallback


def append_log(text: str) -> None:
    with STATUS_LOCK:
        STATUS["log"].append(text)
        STATUS["log"] = STATUS["log"][-800:]


def run_command(job: str, args: list[str]) -> None:
    with STATUS_LOCK:
        if STATUS["running"]:
            raise RuntimeError(f"Already running {STATUS['job']}.")
        STATUS.update({"running": True, "job": job, "returncode": None, "log": []})

    def worker() -> None:
        append_log("> " + " ".join(args) + "\n")
        try:
            process = subprocess.Popen(
                args,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                append_log(line)
            code = process.wait()
        except Exception as exc:
            append_log(f"\nERROR: {exc}\n")
            code = 1
        with STATUS_LOCK:
            STATUS["running"] = False
            STATUS["returncode"] = code

    threading.Thread(target=worker, daemon=True).start()


def current_status() -> dict:
    with STATUS_LOCK:
        return dict(STATUS)


def collect_data(trend_output_value: str = "trend_output", crawl_output_value: str = "crawl_output") -> dict:
    trend_output = safe_path(trend_output_value or "trend_output")
    crawl_output = safe_path(crawl_output_value or "crawl_output")
    return {
        "paths": {
            "root": str(ROOT),
            "trend_output": str(trend_output.relative_to(ROOT)),
            "crawl_output": str(crawl_output.relative_to(ROOT)),
            "python": python_executable(),
            "browser_profile": str(ROOT / ".pinterest_browser_profile"),
        },
        "status": current_status(),
        "trend_package": read_json(trend_output / "trend_package.json", {"trends": []}),
        "manifest": read_json(crawl_output / "crawl_manifest.json", {}),
        "hot_product_images": read_json(crawl_output / "hot_product_images.json", []),
        "rejected_images": read_json(crawl_output / "rejected_images.json", []),
        "raw_results": read_json(crawl_output / "raw_results.json", {"results": []}),
        "vision": read_json(crawl_output / "product_vision_analysis.json", {}),
    }


def remove_output_dir(value: str) -> str:
    path = safe_path(value)
    if path == ROOT:
        raise ValueError("Refusing to remove project root.")
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return str(path.relative_to(ROOT))


def html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pinterest Trend Workbench</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <div class="shell">
    <aside class="side">
      <div class="brand">
        <h1>Pinterest Trend Workbench</h1>
        <p>Run trend discovery, crawl candidate images, then inspect accepted and rejected results in one place.</p>
      </div>

      <section class="panel">
        <h2>Trend Finder</h2>
        <div class="field"><label for="niche">Niche</label><input id="niche" value="rug"></div>
        <input id="trendOutput" type="hidden" value="trend_output">
        <input id="trendPackage" type="hidden" value="trend_output/trend_package.json">
        <input id="crawlOutput" type="hidden" value="crawl_output">
        <div class="row">
          <div class="field"><label for="region">Region</label><input id="region" value="US"></div>
          <div class="field"><label for="maxTrends">Max trends</label><input id="maxTrends" type="number" value="20"></div>
        </div>
        <div class="actions"><button id="runTrend" class="primary">Run Trend Finder</button></div>
      </section>

      <section class="panel">
        <h2>Image Crawler</h2>
        <div class="field">
          <label for="provider">Provider</label>
          <select id="provider">
            <option value="pinterest-browser">pinterest-browser</option>
            <option value="bing-images">bing-images</option>
            <option value="auto">auto</option>
            <option value="pinterest-api">pinterest-api</option>
            <option value="pinterest-web">pinterest-web</option>
          </select>
        </div>
        <div class="field"><label for="crawlTrends">Crawl trends</label><input id="crawlTrends" type="number" value="5"></div>
        <div class="row">
          <div class="field"><label for="maxImages">Images/query</label><input id="maxImages" type="number" value="30"></div>
          <div class="field"><label for="maxQueries">Queries/trend</label><input id="maxQueries" type="number" value="6"></div>
        </div>
        <div class="row">
          <div class="field"><label for="maxDownloads">Downloads</label><input id="maxDownloads" type="number" value="200"></div>
          <div class="field"><label for="minImageScore">Min score</label><input id="minImageScore" type="number" value="20"></div>
        </div>
        <div class="row">
          <div class="field"><label for="topImages">Top images</label><input id="topImages" type="number" value="100"></div>
          <div class="field">
            <label for="visionMode">Vision</label>
            <select id="visionMode">
              <option value="auto">auto</option>
              <option value="required">required</option>
              <option value="off">off</option>
            </select>
          </div>
        </div>
        <div class="field">
          <label for="roles">Accepted roles</label>
          <select id="roles">
            <option value="PRIMARY">PRIMARY only</option>
            <option value="PRIMARY SECONDARY">PRIMARY + SECONDARY</option>
          </select>
        </div>
        <div class="row">
          <div class="field"><label for="minVisibility">Min visibility</label><input id="minVisibility" type="number" value="75"></div>
          <div class="field"><label for="minTrendRelevance">Min trend fit</label><input id="minTrendRelevance" type="number" value="70"></div>
        </div>
        <div class="actions">
          <button id="runCrawler" class="primary">Run Crawler</button>
          <button id="loginPinterest">Open Pinterest Login</button>
          <button id="refresh">Refresh Data</button>
          <button id="clearCrawl" class="danger">Clear Crawl Data</button>
          <button id="clearAll" class="danger ghost">Clear All Data</button>
        </div>
      </section>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <h2>Output Review</h2>
          <p class="muted">Accepted images, trend package, query audit, raw URLs, and reject reasons.</p>
        </div>
        <div id="status" class="status">Loading</div>
      </div>

      <section id="metrics" class="metrics"></section>

      <nav class="tabs" aria-label="Views">
        <button class="tab active" data-tab="accepted">Accepted</button>
        <button class="tab" data-tab="trends">Trends</button>
        <button class="tab" data-tab="queries">Queries</button>
        <button class="tab" data-tab="rejected">Rejected</button>
        <button class="tab" data-tab="raw">Raw URLs</button>
      </nav>

      <section id="content" class="section active"></section>

      <section class="panel">
        <h2>Run Log</h2>
        <pre id="log" class="log"></pre>
      </section>
    </main>
  </div>
  <script src="/static/app.js"></script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(html())
            return
        if parsed.path == "/api/data":
            params = parse_qs(parsed.query)
            try:
                self.send_json(
                    collect_data(
                        params.get("trend_output", ["trend_output"])[0],
                        params.get("crawl_output", ["crawl_output"])[0],
                    )
                )
            except ValueError as exc:
                self.send_error(400, str(exc))
            return
        if parsed.path == "/file":
            params = parse_qs(parsed.query)
            try:
                path = safe_path(params.get("path", [""])[0])
                if not path.exists() or not path.is_file():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception:
                self.send_error(403)
            return
        if parsed.path.startswith("/static/"):
            try:
                path = (UI_DIR / parsed.path.removeprefix("/static/")).resolve()
                if UI_DIR not in path.parents or not path.exists() or not path.is_file():
                    self.send_error(404)
                    return
                self.send_text(path.read_text(encoding="utf-8"), mimetypes.guess_type(str(path))[0] or "text/plain")
            except Exception:
                self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

        try:
            if self.path == "/api/clear-data":
                with STATUS_LOCK:
                    if STATUS["running"]:
                        raise RuntimeError(f"Cannot clear data while {STATUS['job']} is running.")

                scope = str(payload.get("scope") or "crawl")
                removed = []
                if scope in {"crawl", "all"}:
                    removed.append(remove_output_dir(str(payload.get("crawl_output") or "crawl_output")))
                if scope in {"trend", "all"}:
                    removed.append(remove_output_dir(str(payload.get("trend_output") or "trend_output")))

                with STATUS_LOCK:
                    STATUS["log"] = [f"Cleared: {', '.join(removed) if removed else 'nothing'}\n"]
                    STATUS["returncode"] = 0
                self.send_json({"ok": True, "removed": removed})
                return
            if self.path == "/api/run/trends":
                args = [
                    python_executable(),
                    "pinterest_trend_finder.py",
                    "--niche",
                    str(payload.get("niche") or "rug"),
                    "--region",
                    str(payload.get("region") or "US"),
                    "--output",
                    str(payload.get("output") or "trend_output"),
                    "--max-trends",
                    str(payload.get("max_trends") or "20"),
                    "--verbose",
                ]
                run_command("trend finder", args)
                self.send_json({"ok": True})
                return
            if self.path == "/api/run/browser-login":
                args = [
                    python_executable(),
                    "pinterest_browser_login.py",
                    "--timeout",
                    str(payload.get("timeout") or "600"),
                ]
                run_command("pinterest browser login", args)
                self.send_json({"ok": True})
                return
            if self.path == "/api/run/crawler":
                roles = str(payload.get("accepted_product_roles") or "PRIMARY").split()
                args = [
                    python_executable(),
                    "hot_image_crawler.py",
                    "--input",
                    str(payload.get("input") or "trend_output/trend_package.json"),
                    "--provider",
                    str(payload.get("provider") or "bing-images"),
                    "--max-images-per-query",
                    str(payload.get("max_images_per_query") or "30"),
                    "--max-trends",
                    str(payload.get("max_trends") or "5"),
                    "--max-downloads",
                    str(payload.get("max_downloads") or "200"),
                    "--max-queries-per-trend",
                    str(payload.get("max_queries_per_trend") or "6"),
                    "--top-images",
                    str(payload.get("top_images") or "100"),
                    "--min-image-score",
                    str(payload.get("min_image_score") or "20"),
                    "--vision-mode",
                    str(payload.get("vision_mode") or "auto"),
                    "--accepted-product-roles",
                    *(roles or ["PRIMARY"]),
                    "--min-product-visibility",
                    str(payload.get("min_product_visibility") or "75"),
                    "--min-trend-relevance",
                    str(payload.get("min_trend_relevance") or "70"),
                    "--output",
                    str(payload.get("output") or "crawl_output"),
                    "--verbose",
                ]
                run_command("image crawler", args)
                self.send_json({"ok": True})
                return
            self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=409)


def main() -> int:
    port = 8787
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Pinterest Trend Workbench: http://127.0.0.1:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
