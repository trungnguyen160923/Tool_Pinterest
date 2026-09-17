from __future__ import annotations

import base64
import json
import importlib
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
import streamlit as st

import trend_tool.config as config_module
import trend_tool.comparison as comparison_module
import trend_tool.pipeline as pipeline_module
import trend_tool.settings as settings_module
import trend_tool.task4_adapter as task4_adapter_module
import trend_tool.task5_adapter as task5_adapter_module
import trend_tool.crawler as crawler_module
import trend_tool.artwork_generation as artwork_generation_module
import trend_tool.printability as printability_module
import trend_tool.mockup_profile as mockup_profile_module
import trend_tool.template_mockup as template_mockup_module
import trend_tool.report as report_module


config_module = importlib.reload(config_module)
comparison_module = importlib.reload(comparison_module)
settings_module = importlib.reload(settings_module)
task4_adapter_module = importlib.reload(task4_adapter_module)
crawler_module = importlib.reload(crawler_module)
task5_adapter_module = importlib.reload(task5_adapter_module)
artwork_generation_module = importlib.reload(artwork_generation_module)
printability_module = importlib.reload(printability_module)
mockup_profile_module = importlib.reload(mockup_profile_module)
template_mockup_module = importlib.reload(template_mockup_module)
report_module = importlib.reload(report_module)
pipeline_module = importlib.reload(pipeline_module)

PipelineConfig = config_module.PipelineConfig
ProductTarget = config_module.ProductTarget
product_preset = config_module.product_preset
run_pipeline = pipeline_module.run_pipeline
PipelineCancelled = pipeline_module.PipelineCancelled
env = settings_module.env
env_int = settings_module.env_int
load_tool_env = settings_module.load_tool_env
task5_token_path_from_env = settings_module.task5_token_path_from_env
build_comparison_rows = comparison_module.build_comparison_rows


load_tool_env()


def start_pipeline_run(config: PipelineConfig) -> dict[str, object]:
    cancel_event = threading.Event()
    state: dict[str, object] = {
        "cancel_event": cancel_event,
        "logs": [],
        "status": "running",
        "result": None,
        "error": None,
    }

    def worker() -> None:
        def progress(message: str) -> None:
            logs = state["logs"]
            assert isinstance(logs, list)
            logs.append(f"{datetime.now().strftime('%H:%M:%S')} | {message}")

        try:
            state["result"] = run_pipeline(config, progress=progress, cancel_event=cancel_event)
            state["status"] = "complete"
        except PipelineCancelled as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled"
        except Exception as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled" if cancel_event.is_set() else "failed"

    thread = threading.Thread(target=worker, name="trend-product-pipeline", daemon=True)
    state["thread"] = thread
    thread.start()
    return state

st.set_page_config(page_title="Trend Product Tool", layout="wide")

st.title("Trend Product Tool")

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def standalone_output_root() -> Path:
    """Keep the copied UI from accidentally writing into the old project."""
    configured = env("TREND_PRODUCT_OUTPUT", "").strip()
    if configured.replace("\\", "/").rstrip("/") == "trend_product_tool/output":
        return TOOL_ROOT / "output"
    return Path(configured) if configured else TOOL_ROOT / "output"

def read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def list_run_dirs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    runs = [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def format_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def image_files(folder: Path, pattern: str = "*") -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def ai_background_files(run_dir: Path) -> list[Path]:
    lifestyle = image_files(run_dir / "lifestyle_mockups", "*.png")
    if lifestyle:
        return lifestyle
    root = run_dir / "task4_mockups"
    if not root.exists():
        return []
    outputs = sorted(root.rglob("*semantic_strict_output.png"))
    if not outputs:
        outputs = sorted(root.rglob("*deterministic_composite.png"))
    return [path for path in outputs if path.is_file()]


def production_final_files(run_dir: Path, pattern: str) -> list[Path]:
    return image_files(run_dir / "final_print", pattern)


def design_files(run_dir: Path) -> list[Path]:
    return image_files(run_dir / "artwork_designs") + image_files(run_dir / "designs")


def launch_pinterest_browser_login() -> Path:
    script = TOOL_ROOT / "pinterest" / "pinterest_browser_login.py"
    if not script.exists():
        raise RuntimeError(f"Pinterest login helper not found: {script}")
    log_path = TOOL_ROOT / "output" / "pinterest_browser_login.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), "--timeout", "600"]
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return log_path


def pinterest_oauth_config() -> dict[str, str]:
    return {
        "client_id": env("PINTEREST_APP_ID", "").strip(),
        "client_secret": env("PINTEREST_APP_SECRET", "").strip(),
        "scopes": env("PINTEREST_SCOPES", "boards:read,pins:read,user_accounts:read").strip(),
        "redirect_uri": env("PINTEREST_REDIRECT_URI", "http://localhost/").strip(),
    }


def pinterest_authorize_url(client_id: str, redirect_uri: str, scopes: str, state: str) -> str:
    return "https://www.pinterest.com/oauth/?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scopes,
            "state": state,
        }
    )


def extract_pinterest_oauth_code(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.query:
        code = parse_qs(parsed.query).get("code", [""])[0]
        if code:
            return code.strip()
    return text


def exchange_pinterest_oauth_code(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, object]:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    response = requests.post(
        "https://api.pinterest.com/v5/oauth/token",
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "continuous_refresh": "true",
        },
        timeout=30,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}
    if not 200 <= response.status_code < 300:
        preview = json.dumps(payload, ensure_ascii=False, default=str)[:1200]
        raise RuntimeError(f"Pinterest OAuth token exchange failed ({response.status_code}): {preview}")
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("Pinterest OAuth response did not include access_token.")
    issued_at = time.time()
    payload.setdefault("issued_at", issued_at)
    if payload.get("expires_in") and not payload.get("access_token_expires_at"):
        payload["access_token_expires_at"] = issued_at + float(payload["expires_in"])
    return payload


def save_pinterest_oauth_tokens(tokens: dict[str, object]) -> Path:
    path = TOOL_ROOT / ".pinterest_oauth_tokens.json"
    path.write_text(json.dumps(tokens, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def render_pinterest_api_token_ui() -> None:
    cfg = pinterest_oauth_config()
    token_path = TOOL_ROOT / ".pinterest_oauth_tokens.json"
    with st.expander("Pinterest API token", expanded=False):
        st.caption("Use this for Auto trend discovery. Browser login is separate and only used for crawling images.")
        st.write(
            {
                "app_id_configured": bool(cfg["client_id"]),
                "app_secret_configured": bool(cfg["client_secret"]),
                "scopes": cfg["scopes"],
                "redirect_uri": cfg["redirect_uri"],
                "token_file": str(token_path),
                "token_file_exists": token_path.exists(),
            }
        )
        if not cfg["client_id"] or not cfg["client_secret"]:
            st.warning("Set PINTEREST_APP_ID and PINTEREST_APP_SECRET in trend_product_tool\\.env first.")
            return
        if "pinterest_oauth_state" not in st.session_state:
            st.session_state["pinterest_oauth_state"] = secrets.token_urlsafe(16)
        auth_url = pinterest_authorize_url(
            cfg["client_id"],
            cfg["redirect_uri"],
            cfg["scopes"],
            st.session_state["pinterest_oauth_state"],
        )
        st.link_button("Open Pinterest API authorization", auth_url)
        st.caption("After approving, paste the redirected URL or only the code value below.")
        code_or_url = st.text_area(
            "Pinterest callback URL or code",
            value="",
            height=96,
            placeholder="http://localhost/?code=...&state=...",
        )
        if st.button("Exchange and save Pinterest API token"):
            code = extract_pinterest_oauth_code(code_or_url)
            if not code:
                st.error("No OAuth code found.")
                return
            try:
                tokens = exchange_pinterest_oauth_code(
                    client_id=cfg["client_id"],
                    client_secret=cfg["client_secret"],
                    code=code,
                    redirect_uri=cfg["redirect_uri"],
                )
                saved_path = save_pinterest_oauth_tokens(tokens)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved Pinterest API token: {saved_path}")
                st.caption("Run Auto trend discovery again. If Trends still fails, the token/app likely lacks Trends API permission.")


def run_summary(run_dir: Path) -> dict[str, object]:
    config = read_json(run_dir / "config.json")
    stage_manifest = read_json(run_dir / "stage_manifest.json")
    crawl_manifest = read_json(run_dir / "task5_crawl" / "crawl_manifest.json")
    target = {}
    if isinstance(config, dict) and isinstance(config.get("target"), dict):
        target = config["target"]
    counts = {}
    errors = []
    if isinstance(crawl_manifest, dict):
        counts = crawl_manifest.get("counts") if isinstance(crawl_manifest.get("counts"), dict) else {}
        audit = crawl_manifest.get("discovery_audit") if isinstance(crawl_manifest.get("discovery_audit"), dict) else {}
        errors = audit.get("errors") if isinstance(audit.get("errors"), list) else []
    final_pngs = production_final_files(run_dir, "*.png")
    final_jpgs = production_final_files(run_dir, "*.jpg")
    mockups = image_files(run_dir / "mockups")
    ai_backgrounds = ai_background_files(run_dir)
    product_cutouts = image_files(run_dir / "product_cutouts", "*.png")
    product_cutouts_white = image_files(run_dir / "product_cutouts_white", "*.png")
    designs = design_files(run_dir)
    status = "ok" if final_pngs else "empty"
    if errors and not final_pngs:
        status = "crawl_failed"
    if (
        final_pngs
        and isinstance(config, dict)
        and config.get("task4_mockup_engine") in {"task4_ai", "template_ai", "direct_ai", "blender_3d"}
        and not isinstance(stage_manifest, dict)
    ):
        status = "background_partial" if ai_backgrounds else "incomplete"
    if isinstance(stage_manifest, dict):
        background_results = stage_manifest.get("task4_results")
        if isinstance(background_results, list) and any(isinstance(item, dict) and item.get("status") == "failed" for item in background_results):
            status = "background_failed"
        template_results = stage_manifest.get("template_mockup_records")
        if isinstance(template_results, list) and any(isinstance(item, dict) and item.get("status") == "failed" for item in template_results):
            status = "background_partial" if ai_backgrounds else "background_failed"
    return {
        "run": run_dir.name,
        "status": status,
        "modified": format_mtime(run_dir),
        "product": target.get("name", ""),
        "size": f"{target.get('width_px', '')}x{target.get('height_px', '')}".strip("x"),
        "raw": int(counts.get("raw_results") or 0),
        "hot": int(counts.get("hot_product_images") or 0),
        "errors": len(errors),
        "final_png": len(final_pngs),
        "final_jpg": len(final_jpgs),
        "mockups": len(mockups),
        "ai_background": len(ai_backgrounds),
        "product_cutouts": len(product_cutouts) + len(product_cutouts_white),
        "designs": len(designs),
    }


def render_image_grid(paths: list[Path], columns: int = 4) -> None:
    if not paths:
        st.info("No images found.")
        return
    cols = st.columns(columns)
    for index, image_path in enumerate(paths):
        with cols[index % columns]:
            st.image(str(image_path), caption=image_path.name, width="stretch")


def render_paginated_image_grid(
    paths: list[Path],
    *,
    default_page_size: int,
    columns: int = 4,
    key_prefix: str,
) -> None:
    if not paths:
        st.info("No images found.")
        return

    page_size_options = [12, 24, 36, 48, 72, 100, 200]
    default_page_size = max(1, int(default_page_size))
    if default_page_size not in page_size_options:
        page_size_options.append(default_page_size)
        page_size_options = sorted(set(page_size_options))

    controls = st.columns([1, 2, 2])
    with controls[0]:
        page_size = st.selectbox(
            "Images / page",
            page_size_options,
            index=page_size_options.index(default_page_size),
            key=f"{key_prefix}_page_size",
        )

    page_count = max(1, (len(paths) + int(page_size) - 1) // int(page_size))
    with controls[1]:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"{key_prefix}_page",
        )
    with controls[2]:
        st.write("")
        st.caption(f"{len(paths)} image(s), {page_count} page(s)")

    start = (int(page) - 1) * int(page_size)
    end = min(start + int(page_size), len(paths))
    st.caption(f"Showing {start + 1}-{end} / {len(paths)} images.")
    render_image_grid(paths[start:end], columns=columns)


def render_comparison_view(run_dir: Path, preview_limit: int) -> None:
    stage_manifest = read_json(run_dir / "stage_manifest.json")
    rows = build_comparison_rows(run_dir, stage_manifest if isinstance(stage_manifest, dict) else None)
    if not rows:
        st.info("No comparison rows found for this run.")
        return

    page_size_options = [5, 10, 20, 50]
    default_page_size = min(max(1, int(preview_limit)), 10)
    if default_page_size not in page_size_options:
        page_size_options.append(default_page_size)
        page_size_options = sorted(set(page_size_options))
    controls = st.columns([1, 2, 2])
    with controls[0]:
        page_size = st.selectbox(
            "Rows / page",
            page_size_options,
            index=page_size_options.index(default_page_size),
            key=f"{run_dir.name}_compare_page_size",
        )
    page_count = max(1, (len(rows) + int(page_size) - 1) // int(page_size))
    with controls[1]:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"{run_dir.name}_compare_page",
        )
    with controls[2]:
        show_transparent = st.checkbox(
            "Use transparent cutout",
            value=False,
            key=f"{run_dir.name}_compare_transparent",
        )

    start = (int(page) - 1) * int(page_size)
    end = min(start + int(page_size), len(rows))
    st.caption(f"Showing {start + 1}-{end} / {len(rows)} comparison row(s).")
    for row in rows[start:end]:
        title = f"#{row.index} {row.product_label or 'product'} | {row.status or 'unknown'}"
        if row.reason:
            title = f"{title} | {row.reason}"
        with st.container(border=True):
            st.caption(title)
            cols = st.columns(3)
            render_comparison_cell(cols[0], "Pinterest source", row.source_path)
            cutout = row.cutout_path if show_transparent else (row.cutout_white_path or row.cutout_path)
            render_comparison_cell(cols[1], "Product cutout", cutout)
            render_comparison_cell(cols[2], "Final print", row.final_print_path)
            backgrounds = list(row.ai_background_paths)
            st.write(f"AI backgrounds ({len(backgrounds)})")
            if backgrounds:
                background_cols = st.columns(min(4, len(backgrounds)))
                for index, background in enumerate(backgrounds):
                    render_comparison_cell(background_cols[index % len(background_cols)], f"View {index + 1}", background)
            else:
                st.warning("missing")


def render_comparison_cell(column, label: str, path: Path | None) -> None:
    with column:
        st.write(label)
        if path is not None and path.exists():
            st.image(str(path), caption=path.name, width="stretch")
        else:
            st.warning("missing")


def render_run_history(output_root: Path, preview_limit: int) -> None:
    runs = list_run_dirs(output_root)
    st.subheader("Run History")
    st.caption(str(output_root))
    if not runs:
        st.info("No previous runs found.")
        return

    summaries = [run_summary(run_dir) for run_dir in runs]
    run_by_name = {run_dir.name: run_dir for run_dir in runs}
    labels = {
        str(summary["run"]): (
            f"{summary['run']} | {summary['modified']} | {summary['status']} | "
            f"{summary['product']} {summary['size']} | PNG {summary['final_png']} | "
            f"AI bg {summary['ai_background']} | mockups {summary['mockups']}"
        )
        for summary in summaries
    }
    current_run = st.session_state.get("history_selected_run")
    if current_run not in run_by_name:
        current_run = runs[0].name

    selected_run = st.radio(
        "Open run",
        options=[run_dir.name for run_dir in runs],
        index=[run_dir.name for run_dir in runs].index(current_run),
        format_func=lambda name: labels.get(name, name),
        key="history_selected_run",
    )
    run_dir = run_by_name[selected_run]
    st.success(f"Viewing: {run_dir}")

    with st.expander("Run table", expanded=False):
        st.caption("This table is read-only. Use Open run above to switch runs.")
        st.dataframe(summaries, width="stretch", hide_index=True)

    final_pngs = production_final_files(run_dir, "*.png")
    final_jpgs = production_final_files(run_dir, "*.jpg")
    mockups = image_files(run_dir / "mockups")
    ai_backgrounds = ai_background_files(run_dir)
    product_cutouts = image_files(run_dir / "product_cutouts", "*.png")
    product_cutouts_white = image_files(run_dir / "product_cutouts_white", "*.png")
    designs = design_files(run_dir)
    enhanced = image_files(run_dir / "enhanced")
    cropped = image_files(run_dir / "cropped")
    report_path = run_dir / "report.html"
    crawl_manifest = read_json(run_dir / "task5_crawl" / "crawl_manifest.json")

    metric_cols = st.columns(7)
    metric_cols[0].metric("Final PNG", len(final_pngs))
    metric_cols[1].metric("CMYK JPG", len(final_jpgs))
    metric_cols[2].metric("AI Background", len(ai_backgrounds))
    metric_cols[3].metric("Cutouts", len(product_cutouts) + len(product_cutouts_white))
    metric_cols[4].metric("Mockups", len(mockups))
    metric_cols[5].metric("Designs", len(designs))
    metric_cols[6].metric("Enhanced", len(enhanced))

    if not final_pngs:
        st.warning("This run has no final images.")
        render_empty_run_reason(crawl_manifest, run_dir)

    section = st.radio(
        "Run section",
        ["Compare", "Final PNG", "AI Background", "Product Cutouts", "Mockups", "Designs", "Enhanced", "Cropped", "Files", "Config"],
        horizontal=True,
    )
    if section == "Compare":
        render_comparison_view(run_dir, preview_limit)
    elif section == "AI Background":
        render_paginated_image_grid(ai_backgrounds, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_ai_background")
    elif section == "Product Cutouts":
        cutout_view = st.radio(
            "Cutout view",
            ["Transparent PNG", "White background"],
            horizontal=True,
            key=f"{run_dir.name}_cutout_view",
        )
        selected_cutouts = product_cutouts if cutout_view == "Transparent PNG" else product_cutouts_white
        render_paginated_image_grid(selected_cutouts, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_product_cutouts")
    elif section == "Mockups":
        render_paginated_image_grid(mockups, default_page_size=preview_limit, columns=4, key_prefix=f"{run_dir.name}_mockups")
    elif section == "Final PNG":
        render_paginated_image_grid(final_pngs, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_final_png")
    elif section == "Designs":
        render_paginated_image_grid(designs, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_designs")
    elif section == "Enhanced":
        render_paginated_image_grid(enhanced, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_enhanced")
    elif section == "Cropped":
        render_paginated_image_grid(cropped, default_page_size=preview_limit, columns=3, key_prefix=f"{run_dir.name}_cropped")
    elif section == "Files":
        st.write("Run folder")
        st.code(str(run_dir))
        if report_path.exists():
            st.write("Report")
            st.code(str(report_path))
            st.download_button(
                "Download report.html",
                data=report_path.read_text(encoding="utf-8", errors="ignore"),
                file_name=f"{run_dir.name}_report.html",
                mime="text/html",
            )
        final_files = final_pngs + final_jpgs
        if final_files:
            st.write("Final print files")
            st.dataframe(
                [{"name": path.name, "path": str(path), "size_mb": round(path.stat().st_size / 1024 / 1024, 2)} for path in final_files],
                width="stretch",
                hide_index=True,
            )
    else:
        config = read_json(run_dir / "config.json")
        stage_manifest = read_json(run_dir / "stage_manifest.json")
        trend_package = read_json(run_dir / "task5_trends" / "trend_package.json")
        st.write("config.json")
        st.json(config or {})
        st.write("stage_manifest.json")
        st.json(stage_manifest or {})
        if trend_package:
            st.write("trend_package.json")
            st.json(trend_package)


def render_empty_run_reason(crawl_manifest: object | None, run_dir: Path) -> None:
    if not isinstance(crawl_manifest, dict):
        st.caption("No crawl manifest found. The run likely stopped before crawling completed.")
        return
    counts = crawl_manifest.get("counts") if isinstance(crawl_manifest.get("counts"), dict) else {}
    audit = crawl_manifest.get("discovery_audit") if isinstance(crawl_manifest.get("discovery_audit"), dict) else {}
    errors = audit.get("errors") if isinstance(audit.get("errors"), list) else []
    st.write(
        {
            "raw_results": counts.get("raw_results", 0),
            "image_candidates": counts.get("image_candidates", 0),
            "hot_product_images": counts.get("hot_product_images", 0),
            "rejected_images": counts.get("rejected_images", 0),
        }
    )
    if errors:
        first_error = errors[0] if isinstance(errors[0], dict) else {}
        st.error("Crawler did not collect images. The first crawl error points to Pinterest browser login or HTTPS access.")
        st.code(str(first_error.get("error", first_error))[:4000], language="text")
    log_path = run_dir / "task5_crawl" / "task5_image_crawler.log"
    if log_path.exists():
        st.caption(f"Full crawler log: {log_path}")

with st.sidebar:
    st.subheader("Main")
    product_options = ["rug", "blanket", "custom"]
    product = st.selectbox("Product", product_options)
    preset = product_preset(product)
    task5_provider = "pinterest-browser"
    output_root_text = str(standalone_output_root())
    width = preset.width_px
    height = preset.height_px
    dpi = preset.dpi
    crop_mode = "cover"
    remove_white = False
    export_cmyk = True
    enhancement_mode = "task2_local"
    artwork_image_size = env("GEMINI_ARTWORK_IMAGE_SIZE", "2K")
    dedupe_threshold = 6
    preview_designs = env_int("PREVIEW_DESIGNS", 72)
    task5_vision_mode = "auto"
    task5_refresh_vision_cache = True
    trend_region = env("PINTEREST_REGION", "US")
    trend_type = "growing"
    trend_interest = ""
    trend_keyword_limit = 50
    trend_max_trends = 20
    trend_min_fit = 35.0
    trend_max_queries = 6
    task5_token_path_text = ""
    gemini_backend = env("GEMINI_BACKEND", "auto")
    gemini_model = env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash")

    niche_default = env("PINTEREST_NICHE", "").strip()
    trend_niche = st.text_input(
        "Trend niche",
        value=niche_default,
        placeholder="Examples: rug, halloween decor, coastal grandmother decor",
    ).strip()
    st.caption("Pinterest uses this to find and rank relevant trends. Browser crawl uses the original hot-trend keyword returned by the API.")

    if st.button("Open Pinterest login", help="Opens Chromium and saves the profile used by pinterest-browser."):
        try:
            login_log_path = launch_pinterest_browser_login()
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success("Pinterest login window opened. Finish login in Chromium, then rerun Find and build.")
            st.caption(f"Login log: {login_log_path}")

    render_pinterest_api_token_ui()

    desired_output_count = st.slider(
        "Desired output images",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
    background_variants = st.slider(
        "AI backgrounds per product",
        min_value=1,
        max_value=4,
        value=4,
        step=1,
    )
    # Keep the user-facing count about deliverables; the crawler needs a wider
    # source pool because quality gates intentionally reject most references.
    crawl_budget = min(250, max(30, int(desired_output_count) * 10))
    task5_max_downloads = crawl_budget
    task5_max_images_per_query = max(12, min(30, int(desired_output_count) * 4))
    task5_max_crawl_trends = 5
    task5_top_images = int(task5_max_downloads)

    design_mode = "ai_artwork"
    st.caption("Crawled artwork references must pass printability checks before output.")

    with st.expander("Advanced settings", expanded=False):
        output_root_text = st.text_input("Output folder", value=output_root_text)
        artwork_image_size = st.selectbox("Artwork image size", ["1K", "2K", "4K"], index=["1K", "2K", "4K"].index(artwork_image_size) if artwork_image_size in {"1K", "2K", "4K"} else 1)
        trend_region = st.text_input("Trend region", value=trend_region)
        trend_type_options = ["growing", "monthly", "yearly", "seasonal"]
        trend_type = st.selectbox("Trend type", trend_type_options, index=trend_type_options.index(env("PINTEREST_TREND_TYPE", "growing")) if env("PINTEREST_TREND_TYPE", "growing") in trend_type_options else 0)
        task5_token_default = task5_token_path_from_env()
        task5_token_path_text = st.text_input("Pinterest API token path", value=str(task5_token_default or ""))
        if product == "custom":
            width = st.number_input("Width px", min_value=512, max_value=20000, value=int(width), step=100)
            height = st.number_input("Height px", min_value=512, max_value=20000, value=int(height), step=100)
            dpi = st.number_input("DPI", min_value=72, max_value=600, value=int(dpi), step=1)

target = ProductTarget(
    name=product,
    width_px=int(width),
    height_px=int(height),
    dpi=int(dpi),
    prefer_cmyk=True,
    allow_custom_shape=product == "custom",
)
left, middle = st.columns([1, 1])
with left:
    st.metric("Workflow", "Trend to product")
with middle:
    st.metric("Target", f"{target.width_px} x {target.height_px}")

active_run = st.session_state.get("trend_product_active_run")
is_running = isinstance(active_run, dict) and active_run.get("status") == "running"
run_column, stop_column = st.columns(2)
with run_column:
    run_clicked = st.button("Find and build", type="primary", disabled=bool(is_running))
with stop_column:
    stop_clicked = st.button("Stop", disabled=not bool(is_running))

if stop_clicked and isinstance(active_run, dict):
    cancel_event = active_run.get("cancel_event")
    if isinstance(cancel_event, threading.Event):
        cancel_event.set()
        st.warning("Stopping the active Find and build run.")

if run_clicked:
    if not trend_niche:
        st.error("Enter a niche before starting Pinterest Trends discovery.")
    else:
        output_root = Path(output_root_text)
        task5_token_path = Path(task5_token_path_text) if task5_token_path_text.strip() else None
        config = PipelineConfig(
            target=target,
            output_root=output_root,
            workflow_mode="trend_to_product",
            trend_niche=trend_niche,
            trend_region=trend_region,
            trend_type=trend_type,
            trend_interest=trend_interest,
            trend_keyword_limit=int(trend_keyword_limit),
            trend_max_trends=int(trend_max_trends),
            trend_min_semantic_fit=float(trend_min_fit),
            trend_max_queries_per_trend=int(trend_max_queries),
            desired_output_count=int(desired_output_count),
            gemini_backend=gemini_backend,
            gemini_model=gemini_model,
            task5_token_path=task5_token_path,
            task5_provider=task5_provider,
            task5_max_images_per_query=int(task5_max_images_per_query),
            task5_max_crawl_trends=int(task5_max_crawl_trends),
            task5_max_downloads=int(task5_max_downloads),
            task5_top_images=int(task5_top_images),
            task5_vision_mode=task5_vision_mode,
            task5_refresh_vision_cache=bool(task5_refresh_vision_cache),
            crop_mode=str(crop_mode),
            dedupe_threshold=int(dedupe_threshold),
            remove_white_background=bool(remove_white),
            export_cmyk=bool(export_cmyk),
            design_mode=design_mode,
            artwork_image_size=artwork_image_size,
            enhancement_mode=enhancement_mode,
            task4_mockup_engine="direct_ai",
            task4_ai_limit=int(desired_output_count),
            task4_variants_per_product=int(background_variants),
        )
        active_run = start_pipeline_run(config)
        st.session_state["trend_product_active_run"] = active_run

if isinstance(active_run, dict):
    run_status = str(active_run.get("status") or "running")
    logs = active_run.get("logs")
    if isinstance(logs, list) and logs:
        st.code("\n".join(str(line) for line in logs[-80:]), language="text")

    if run_status == "running":
        st.info("Find and build is running. Stop cancels the current pipeline and Pinterest subprocesses.")
        time.sleep(0.5)
        st.rerun()
    elif run_status == "cancelled":
        st.warning(str(active_run.get("error") or "Find and build was stopped."))
    elif run_status == "failed":
        error_text = str(active_run.get("error") or "")
        st.error(error_text)
        if "pinterest trend discovery failed" in error_text.lower() or "client_credentials token request failed" in error_text.lower():
            st.warning("Pinterest Trends API authentication failed. Check credentials/permissions and retry.")
        else:
            st.warning("The run stopped before completion. Check the log above and the run folder for details.")
    else:
        result = active_run.get("result")
        if result is not None:
            st.success(f"Run folder: {result.run_dir}")
            st.caption(f"Report: {result.report_path}")
            design_previews = [path for path in result.final_images if path.suffix.lower() == ".png"]
            shown_count = min(int(preview_designs), len(design_previews))
            st.caption(f"Design previews: showing {shown_count} / {len(design_previews)} PNG designs. Full print files: {len(result.final_images)} files including RGB PNG and CMYK JPG.")
            if result.mockups:
                st.subheader("Mockups")
                mockup_cols = st.columns(4)
                for index, image_path in enumerate(result.mockups):
                    with mockup_cols[index % 4]:
                        st.image(str(image_path), caption=image_path.name, width="stretch")
            st.subheader("Designs")
            cols = st.columns(3)
            for index, image_path in enumerate(design_previews[:shown_count]):
                with cols[index % 3]:
                    st.image(str(image_path), caption=image_path.name, width="stretch")

st.divider()
render_run_history(Path(output_root_text), int(preview_designs))







