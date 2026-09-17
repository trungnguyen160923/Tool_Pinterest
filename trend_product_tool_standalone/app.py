from __future__ import annotations

import base64
import dataclasses
from dataclasses import replace
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
import streamlit.components.v1 as components

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
run_crawl_and_review_stage = pipeline_module.run_crawl_and_review_stage
run_production_from_candidates = pipeline_module.run_production_from_candidates
CandidateReviewItem = pipeline_module.CandidateReviewItem
CandidateReviewPackage = pipeline_module.CandidateReviewPackage
PipelineCancelled = pipeline_module.PipelineCancelled
env = settings_module.env
env_int = settings_module.env_int
load_tool_env = settings_module.load_tool_env
task5_token_path_from_env = settings_module.task5_token_path_from_env
build_comparison_rows = comparison_module.build_comparison_rows

load_tool_env()

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


@st.cache_data(max_entries=200)
def get_image_thumbnail_bytes(image_path_str: str, max_size: int = 800) -> bytes:
    """Creates a fast, cached thumbnail in memory to prevent browser websocket payload bloat."""
    from PIL import Image
    import io
    try:
        with Image.open(image_path_str) as img:
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA"):
                img.save(buf, format="PNG", optimize=True)
            else:
                img.convert("RGB").save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception:
        return Path(image_path_str).read_bytes()


def trigger_tab_switch(tab_index: int) -> None:
    """Sets target tab index for Streamlit's native workflow tab state."""
    st.session_state["switch_to_tab"] = tab_index


def safe_download_button(file_path: Path, label: str, key: str, mime: str | None = None) -> None:
    """Renders a direct download button for a file if it exists."""
    if not file_path.exists():
        st.button(f"⚠ Tệp không tìm thấy ({file_path.name})", disabled=True, key=key, use_container_width=True)
        return
    if mime is None:
        s = file_path.suffix.lower()
        if s == ".png":
            mime = "image/png"
        elif s in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif s == ".html":
            mime = "text/html"
        else:
            mime = "application/octet-stream"
    # For small files (< 2 MB), render download_button directly
    file_size = file_path.stat().st_size
    if file_size < 2 * 1024 * 1024:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            st.download_button(
                label=label,
                data=data,
                file_name=file_path.name,
                mime=mime,
                key=key,
                use_container_width=True,
            )
            return
        except Exception as exc:
            st.error(f"Không thể đọc {file_path.name}: {exc}")
            return

    # For large print masters / mockups (>= 2 MB), use on-demand 1-click preparation to keep WebSocket messages fast and prevent React Aria tab drop
    ready_key = f"ready_{key}"
    if not st.session_state.get(ready_key):
        if st.button(label, key=f"prep_{key}", use_container_width=True):
            st.session_state[ready_key] = True
            st.rerun()
    else:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"💾 Bấm Tải Ngay {file_path.name} ({round(file_size/(1024*1024), 1)} MB)",
                data=data,
                file_name=file_path.name,
                mime=mime,
                key=f"dl_act_{key}",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Không thể tải {file_path.name}: {exc}")


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
    with st.expander("Pinterest API Token", expanded=False):
        st.caption("Cấu hình API Token để tự động cào trend. Đăng nhập trình duyệt dành riêng cho tải ảnh.")
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
            st.warning("Thiết lập PINTEREST_APP_ID và PINTEREST_APP_SECRET trong tệp .env.")
            return
        if "pinterest_oauth_state" not in st.session_state:
            st.session_state["pinterest_oauth_state"] = secrets.token_urlsafe(16)
        auth_url = pinterest_authorize_url(
            cfg["client_id"],
            cfg["redirect_uri"],
            cfg["scopes"],
            st.session_state["pinterest_oauth_state"],
        )
        st.link_button("Mở xác thực Pinterest OAuth", auth_url)
        code_or_url = st.text_area(
            "Callback URL hoặc Mã Code Pinterest",
            value="",
            height=80,
            placeholder="http://localhost/?code=...&state=...",
        )
        if st.button("Lưu Pinterest API Token"):
            code = extract_pinterest_oauth_code(code_or_url)
            if not code:
                st.error("Không tìm thấy mã OAuth code.")
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
                st.success(f"Đã lưu Pinterest API token: {saved_path}")


def run_summary(run_dir: Path) -> dict[str, object]:
    config = read_json(run_dir / "config.json")
    stage_manifest = read_json(run_dir / "stage_manifest.json")
    crawl_manifest = read_json(run_dir / "task5_crawl" / "crawl_manifest.json")
    cand_review = read_json(run_dir / "candidate_review.json")
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
    has_candidate_review = cand_review is not None
    candidates_count = len(cand_review.get("candidates", [])) if isinstance(cand_review, dict) else 0

    status = "ok" if final_pngs else "empty"
    if has_candidate_review and not final_pngs:
        if isinstance(cand_review, dict) and cand_review.get("status") == "failed":
            status = "crawl_failed"
        else:
            status = "review_ready"
    elif errors and not final_pngs:
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
        "candidates": candidates_count,
    }


def render_image_grid(paths: list[Path], columns: int = 4) -> None:
    if not paths:
        st.info("Không có hình ảnh nào.")
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
        st.info("Không tìm thấy hình ảnh.")
        return

    page_size_options = [12, 24, 36, 48, 72, 100, 200]
    default_page_size = max(1, int(default_page_size))
    if default_page_size not in page_size_options:
        page_size_options.append(default_page_size)
        page_size_options = sorted(set(page_size_options))

    controls = st.columns([1, 2, 2])
    with controls[0]:
        page_size = st.selectbox(
            "Số ảnh / trang",
            page_size_options,
            index=page_size_options.index(default_page_size),
            key=f"{key_prefix}_page_size",
        )

    page_count = max(1, (len(paths) + int(page_size) - 1) // int(page_size))
    with controls[1]:
        page = st.number_input(
            "Trang",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"{key_prefix}_page",
        )
    with controls[2]:
        st.write("")
        st.caption(f"Tổng cộng {len(paths)} ảnh, {page_count} trang.")

    start = (int(page) - 1) * int(page_size)
    end = min(start + int(page_size), len(paths))
    st.caption(f"Đang hiển thị {start + 1}-{end} / {len(paths)} ảnh.")
    render_image_grid(paths[start:end], columns=columns)


def render_comparison_cell(column, label: str, path: Path | None) -> None:
    with column:
        st.write(label)
        if path is not None and path.exists():
            st.image(get_image_thumbnail_bytes(str(path)), caption=path.name, width="stretch")
        else:
            st.warning("Thiếu tệp")


def render_comparison_view(run_dir: Path, preview_limit: int) -> None:
    stage_manifest = read_json(run_dir / "stage_manifest.json")
    rows = build_comparison_rows(run_dir, stage_manifest if isinstance(stage_manifest, dict) else None)
    if not rows:
        st.info("Chưa có bảng so sánh đối chiếu cho mẻ chạy này.")
        return

    page_size_options = [5, 10, 20, 50]
    default_page_size = min(max(1, int(preview_limit)), 10)
    if default_page_size not in page_size_options:
        page_size_options.append(default_page_size)
        page_size_options = sorted(set(page_size_options))
    controls = st.columns([1, 2, 2])
    with controls[0]:
        page_size = st.selectbox(
            "Số dòng / trang",
            page_size_options,
            index=page_size_options.index(default_page_size),
            key=f"{run_dir.name}_compare_page_size",
        )
    page_count = max(1, (len(rows) + int(page_size) - 1) // int(page_size))
    with controls[1]:
        page = st.number_input(
            "Trang",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"{run_dir.name}_compare_page",
        )
    with controls[2]:
        show_transparent = st.checkbox(
            "Dùng phôi cắt trong suốt (Transparent)",
            value=False,
            key=f"{run_dir.name}_compare_transparent",
        )

    start = (int(page) - 1) * int(page_size)
    end = min(start + int(page_size), len(rows))
    st.caption(f"Đang hiển thị {start + 1}-{end} / {len(rows)} hàng so sánh đối chiếu.")
    for row in rows[start:end]:
        title = f"#{row.index} {row.product_label or 'product'} | {row.status or 'unknown'}"
        if row.reason:
            title = f"{title} | {row.reason}"
        with st.container(border=True):
            st.caption(title)
            cols = st.columns(3)
            render_comparison_cell(cols[0], "1. Ảnh gốc Pinterest", row.source_path)
            cutout = row.cutout_path if show_transparent else (row.cutout_white_path or row.cutout_path)
            render_comparison_cell(cols[1], "2. Phôi cắt sản phẩm", cutout)
            render_comparison_cell(cols[2], "3. Bản in siêu nét 4K", row.final_print_path)
            backgrounds = list(row.ai_background_paths)
            st.write(f"4. Phối cảnh AI phòng khách ({len(backgrounds)} góc nhìn)")
            if backgrounds:
                background_cols = st.columns(min(4, len(backgrounds)))
                for index, background in enumerate(backgrounds):
                    render_comparison_cell(background_cols[index % len(background_cols)], f"Góc nhìn {index + 1}", background)
            else:
                st.warning("Chưa tạo ảnh phối cảnh")


def restore_pipeline_config(raw_cfg: dict, fallback_root: Path) -> PipelineConfig:
    return config_module.restore_pipeline_config(raw_cfg, fallback_root)


def render_empty_run_reason(crawl_manifest: object | None, run_dir: Path) -> None:
    if not isinstance(crawl_manifest, dict):
        st.caption("Không tìm thấy crawl manifest. Mẻ chạy có thể đã dừng trước khi hoàn thành cào ảnh.")
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
        st.error("Crawler không thu thập được ảnh. Vui lòng kiểm tra lại đăng nhập Chromium hoặc quyền Pinterest API.")
        st.code(str(first_error.get("error", first_error))[:4000], language="text")
    log_path = run_dir / "task5_crawl" / "task5_image_crawler.log"
    if log_path.exists():
        st.caption(f"Tệp nhật ký cào: {log_path}")


def start_crawl_and_review_run(config: PipelineConfig) -> dict[str, object]:
    cancel_event = threading.Event()
    state: dict[str, object] = {
        "kind": "crawl_and_review",
        "cancel_event": cancel_event,
        "logs": [],
        "status": "running",
        "package": None,
        "error": None,
    }

    def worker() -> None:
        def progress(message: str) -> None:
            logs = state["logs"]
            assert isinstance(logs, list)
            logs.append(f"{datetime.now().strftime('%H:%M:%S')} | {message}")

        try:
            state["package"] = run_crawl_and_review_stage(config, progress=progress, cancel_event=cancel_event)
            state["status"] = "review_ready"
        except PipelineCancelled as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled"
        except Exception as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled" if cancel_event.is_set() else "failed"

    thread = threading.Thread(target=worker, name="trend-crawl-review", daemon=True)
    state["thread"] = thread
    thread.start()
    return state


def start_production_run(
    selected_items: list,
    config: PipelineConfig,
    run_dir: Path,
) -> dict[str, object]:
    cancel_event = threading.Event()
    state: dict[str, object] = {
        "kind": "production",
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
            state["result"] = run_production_from_candidates(
                selected_items,
                config,
                run_dir=run_dir,
                progress=progress,
                cancel_event=cancel_event,
            )
            state["status"] = "complete"
        except PipelineCancelled as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled"
        except Exception as exc:
            state["error"] = str(exc)
            state["status"] = "cancelled" if cancel_event.is_set() else "failed"

    thread = threading.Thread(target=worker, name="trend-production", daemon=True)
    state["thread"] = thread
    thread.start()
    return state


def start_pipeline_run(config: PipelineConfig) -> dict[str, object]:
    cancel_event = threading.Event()
    state: dict[str, object] = {
        "kind": "pipeline",
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


# -----------------------------------------------------------------------------
# TAB 1: KHÁM PHÁ & CÀO TREND (CRAWL & DISCOVER)
# -----------------------------------------------------------------------------
def render_crawl_tab(
    output_root: Path,
    active_run: dict | None,
    is_running: bool,
    global_settings: dict,
) -> None:
    st.markdown("### 🔍 Bước 1: Khám phá Xu hướng & Cào Hoa văn Pinterest")
    st.caption("Tìm kiếm các xu hướng thiết kế nội thất / dệt may hot nhất trên Pinterest và chấm điểm AI Vision tự động.")

    col_target, col_niche = st.columns([1.2, 1.8])
    with col_target:
        st.markdown("**1. Chọn Loại Sản phẩm (Product Target)**")
        prod_preset_choice = st.radio(
            "Kích thước phôi chuẩn",
            [
                "🛋️ Rug (Thảm sàn) - 4000 x 6400 px (300 DPI)",
                "🛌 Blanket (Chăn ném) - 10000 x 11000 px (300 DPI)",
                "⚙️ Tùy chỉnh kích thước (Custom)",
            ],
            index=0,
            key="tab1_preset_choice",
        )
        if "Blanket" in prod_preset_choice:
            product_name = "blanket"
            preset = product_preset("blanket")
            width, height, dpi = preset.width_px, preset.height_px, preset.dpi
        elif "Custom" in prod_preset_choice:
            product_name = "custom"
            sub_w, sub_h, sub_dpi = st.columns(3)
            width = sub_w.number_input("Rộng (px)", min_value=512, max_value=20000, value=4000, step=100)
            height = sub_h.number_input("Cao (px)", min_value=512, max_value=20000, value=6400, step=100)
            dpi = sub_dpi.number_input("DPI", min_value=72, max_value=600, value=300, step=1)
        else:
            product_name = "rug"
            preset = product_preset("rug")
            width, height, dpi = preset.width_px, preset.height_px, preset.dpi

    with col_niche:
        st.markdown("**2. Định hướng Xu hướng (Trend Niche)**")
        trend_niche = st.text_input(
            "Từ khóa Niche (Trend Niche)",
            value=env("PINTEREST_NICHE", "rug").strip(),
            placeholder="Ví dụ: rug, boho living room, botanical pattern, checkerboard...",
            help="Pinterest sẽ dùng từ khóa này để xếp hạng xu hướng tăng trưởng nhanh nhất.",
            key="tab1_trend_niche_input",
        ).strip()
        st.markdown(
            "💡 **Gợi ý Niche:** `rug` · `boho living room` · `botanical pattern` · `checkerboard` · `moroccan vintage` · `minimalist abstract`"
        )

        with st.expander("⚙️ Tinh chỉnh tham số cào & sản xuất", expanded=False):
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                desired_count = st.slider("Số lượng mẫu cần sản xuất", min_value=1, max_value=20, value=5, step=1, key="tab1_desired_count")
            with c_p2:
                bg_variants = st.slider("Số góc mockup AI / mẫu", min_value=1, max_value=4, value=4, step=1, key="tab1_bg_variants")
            workflow_mode_select = st.radio(
                "Chế độ vận hành",
                ["Interactive (2-Step Review: Khuyên dùng)", "1-Click Auto (Tự động cào và xuất xưởng luôn)"],
                index=0,
                key="tab1_workflow_mode",
                help="Interactive cho phép bạn duyệt và tự tay chọn mẫu hoa văn đẹp nhất ở Bước 2.",
            )

    target = ProductTarget(
        name=product_name,
        width_px=int(width),
        height_px=int(height),
        dpi=int(dpi),
        prefer_cmyk=True,
        allow_custom_shape=product_name == "custom",
    )

    st.divider()
    cta_col1, cta_col2 = st.columns([2, 3])
    with cta_col1:
        if "Interactive" in workflow_mode_select:
            cta_btn_label = "🚀 1. Khởi động Cào & Chấm điểm AI"
            cta_help = "Cào hoa văn từ Pinterest và dùng AI chấm điểm thẩm định độ phẳng và chất lượng in ấn."
        else:
            cta_btn_label = "🚀 Khởi động Cào & Sản xuất Tự động (1-Click Auto)"
            cta_help = "Tự động chạy toàn bộ quy trình từ cào ảnh đến xuất bản in 4K và mockup AI."

        start_btn = st.button(
            cta_btn_label,
            type="primary",
            disabled=bool(is_running),
            use_container_width=True,
            help=cta_help,
            key="tab1_start_btn",
        )

    with cta_col2:
        if is_running:
            stop_btn = st.button("🛑 Dừng Quy trình Đang Chạy", type="secondary", key="tab1_stop_btn")
            if stop_btn and isinstance(active_run, dict):
                cancel_event = active_run.get("cancel_event")
                if isinstance(cancel_event, threading.Event):
                    cancel_event.set()
                    st.warning("Đang gửi lệnh dừng quy trình...")
        else:
            st.caption("Hệ thống sẽ dùng Chromium không giao diện để thu thập ảnh Pinterest, lọc hoa văn độc hại, và phân tích độ phẳng 2D.")

    if start_btn:
        if not trend_niche:
            st.error("Vui lòng nhập từ khóa Niche trước khi khởi động.")
        else:
            crawl_budget = min(250, max(30, int(desired_count) * 10))
            task5_token_path = Path(global_settings["task5_token_path"]) if global_settings.get("task5_token_path") else None
            config = PipelineConfig(
                target=target,
                output_root=output_root,
                workflow_mode="trend_to_product",
                trend_niche=trend_niche,
                trend_region=global_settings["trend_region"],
                trend_type=global_settings["trend_type"],
                trend_interest="",
                trend_keyword_limit=50,
                trend_max_trends=20,
                trend_min_semantic_fit=35.0,
                trend_max_queries_per_trend=6,
                desired_output_count=int(desired_count),
                gemini_backend=global_settings["gemini_backend"],
                gemini_model=global_settings["gemini_model"],
                task5_token_path=task5_token_path,
                task5_provider="pinterest-browser",
                task5_max_images_per_query=max(12, min(30, int(desired_count) * 4)),
                task5_max_crawl_trends=5,
                task5_max_downloads=int(crawl_budget),
                task5_top_images=int(crawl_budget),
                task5_vision_mode="auto",
                task5_refresh_vision_cache=True,
                crop_mode="cover",
                dedupe_threshold=6,
                remove_white_background=False,
                export_cmyk=True,
                design_mode=global_settings["design_mode"],
                artwork_image_size=global_settings["artwork_image_size"],
                enhancement_mode="task2_local",
                task4_mockup_engine="direct_ai",
                task4_ai_limit=int(desired_count),
                task4_variants_per_product=int(bg_variants),
            )
            st.session_state["active_pipeline_config"] = config
            if "Interactive" in workflow_mode_select:
                new_run = start_crawl_and_review_run(config)
            else:
                new_run = start_pipeline_run(config)
            st.session_state["trend_product_active_run"] = new_run
            st.rerun()

    # Progress and real-time logs container
    if isinstance(active_run, dict):
        run_status = str(active_run.get("status") or "")
        run_kind = str(active_run.get("kind") or "")
        logs = active_run.get("logs") or []

        if run_status == "running" and run_kind in {"crawl_and_review", "pipeline"}:
            st.info("⏳ **Bước 1 đang thực thi:** Đang thu thập xu hướng Pinterest và thẩm định AI Vision...")
            pct = 0.20
            if logs:
                last_log = str(logs[-1]).lower()
                if "review" in last_log or "score" in last_log:
                    pct = 0.85
                elif "filter" in last_log or "dedupe" in last_log:
                    pct = 0.70
                elif "download" in last_log or "crawl" in last_log or "collect" in last_log:
                    pct = 0.50
                elif "trend" in last_log or "discovering" in last_log:
                    pct = 0.30
            st.progress(pct, text=f"Đang cào dữ liệu và phân tích hình ảnh ({int(pct * 100)}%)...")
        elif run_status == "running" and run_kind == "production":
            st.info("⏳ **Bước 2 (Sản xuất ảnh)** đang thực thi ở Tab 2. Bạn có thể chuyển sang Tab 2 để theo dõi tiến độ.")
        elif run_status in {"review_ready", "review_ready_handled"}:
            st.success("🎉 **Bước 1 hoàn thành:** Đã cào và sàng lọc hoa văn xong! Hãy chuyển sang **Tab 2. Duyệt & Tuyển chọn Hoa văn** để chọn mẫu in.")
            if st.button("👉 Chuyển sang Tab 2 ngay", key="tab1_goto_tab2"):
                st.session_state["switch_to_tab"] = 1
                st.rerun()
        elif run_status == "cancelled":
            st.warning(f"Quy trình đã dừng: {active_run.get('error') or 'Người dùng hủy thao tác.'}")
        elif run_status == "failed":
            st.error(f"Lỗi quy trình: {active_run.get('error')}")

        if logs:
            with st.expander("📋 Nhật ký tiến trình thời gian thực (Console Logs)", expanded=is_running):
                st.code("\n".join(str(l) for l in logs[-80:]), language="text")


# -----------------------------------------------------------------------------
# TAB 2: DUYỆT & TUYỂN CHỌN HOA VĂN (CURATE & REVIEW)
# -----------------------------------------------------------------------------
def render_candidate_review_ui(
    package_data: dict | CandidateReviewPackage,
    config: PipelineConfig | None = None,
    is_running: bool = False,
    key_prefix: str = "review",
) -> None:
    st.markdown("### 🎨 Bước 2: Duyệt & Tuyển chọn Hoa văn (Human-in-the-Loop Review)")
    st.caption("Kiểm định điểm chất lượng AI, độ phẳng 2D và chọn lọc các mẫu hoa văn ưng ý nhất để xuất bản in 4K & Mockup.")

    if isinstance(package_data, CandidateReviewPackage):
        run_dir = package_data.run_dir
        candidates = package_data.candidates
    elif isinstance(package_data, dict):
        run_dir = Path(str(package_data.get("run_dir") or "."))
        candidates_raw = package_data.get("candidates") or []
        candidates = []
        valid_fields = {f.name for f in dataclasses.fields(CandidateReviewItem)}
        for item in candidates_raw:
            if isinstance(item, dict):
                clean_item = {k: v for k, v in item.items() if k in valid_fields}
                candidates.append(CandidateReviewItem(**clean_item))
            elif isinstance(item, CandidateReviewItem):
                candidates.append(item)
    else:
        st.info("Không có dữ liệu hoa văn hợp lệ.")
        return

    if not candidates:
        st.info("Không tìm thấy hoa văn nào trong danh sách thẩm định.")
        return

    direct_count = sum(1 for c in candidates if c.is_direct_printable)
    pattern_count = sum(1 for c in candidates if "pattern" in c.classification.lower())

    prefix = f"{key_prefix}_{run_dir.name}"
    for c in candidates:
        check_key = f"sel_{prefix}_{c.image_id}"
        if check_key not in st.session_state:
            st.session_state[check_key] = bool(c.recommended)

    selected_items = [
        c for c in candidates
        if st.session_state.get(f"sel_{prefix}_{c.image_id}", c.recommended)
    ]

    # TOP PROMINENT ACTION BAR (Pinned right at top of Tab 2)
    with st.container(border=True):
        st.caption("⚡ **Thanh Hành Động Sản Xuất Nhanh (Top Action Bar)**")
        top_col1, top_col2, top_col3, top_col4 = st.columns([1.5, 1.2, 1.2, 2.2])
        with top_col1:
            st.metric("Đã chọn sản xuất", f"{len(selected_items)} / {len(candidates)} mẫu")
        with top_col2:
            st.metric("Chuẩn In Trực tiếp", f"{direct_count} mẫu")
        with top_col3:
            st.metric("Hoa văn phẳng (2D)", f"{pattern_count} mẫu")
        with top_col4:
            step2_design_mode = st.selectbox(
                "Chế độ thiết kế Bước 2",
                ["direct", "ai_artwork"],
                format_func=lambda x: "Direct Print (In sắc nét hoa văn gốc)" if x == "direct" else "AI Artwork (Gemini vẽ lại)",
                index=0 if getattr(config, "design_mode", "direct") == "direct" else 1,
                key=f"top_mode_{prefix}",
                help="Direct Print tăng nét hoa văn gốc Pinterest. AI Artwork nhờ Gemini vẽ lại hoa văn.",
            )
            top_produce_clicked = st.button(
                f"🚀 2. Produce Selected Images ({len(selected_items)} mẫu)",
                type="primary",
                disabled=len(selected_items) == 0 or bool(is_running),
                key=f"top_btn_produce_{prefix}",
                use_container_width=True,
            )

        # Quick Bulk Selection Buttons directly inside Top Action Bar
        act_row = st.columns([1.5, 1.5, 1.5, 2.5])
        with act_row[0]:
            if st.button("✓ Chọn tất cả", key=f"bar_sel_all_{prefix}", use_container_width=True):
                for c in candidates:
                    st.session_state[f"sel_{prefix}_{c.image_id}"] = True
                st.rerun()
        with act_row[1]:
            if st.button("✗ Bỏ chọn tất cả", key=f"bar_desel_all_{prefix}", use_container_width=True):
                for c in candidates:
                    st.session_state[f"sel_{prefix}_{c.image_id}"] = False
                st.rerun()
        with act_row[2]:
            if st.button("⭐ Top AI gợi ý", key=f"bar_sel_top_{prefix}", use_container_width=True):
                for c in candidates:
                    st.session_state[f"sel_{prefix}_{c.image_id}"] = bool(c.recommended)
                st.rerun()
        with act_row[3]:
            st.caption(f"Đã chọn {len(selected_items)} mẫu sẵn sàng xuất file in 4K & Mockup AI.")

    # Real-time Production Progress & Console Logs in Tab 2
    active_run = st.session_state.get("trend_product_active_run")
    if is_running and isinstance(active_run, dict) and active_run.get("kind") == "production":
        with st.container(border=True):
            st.info("⏳ **Bước 2 đang thực thi:** Đang xử lý độ phân giải 4K CMYK, tách phôi sản phẩm và dựng mockup AI...")
            prod_logs = active_run.get("logs") or []
            prod_pct = 0.20
            if prod_logs:
                last_log = str(prod_logs[-1]).lower()
                if "mockup" in last_log or "background" in last_log:
                    prod_pct = 0.85
                elif "print" in last_log or "cmyk" in last_log or "final" in last_log:
                    prod_pct = 0.65
                elif "enhance" in last_log or "design" in last_log or "crop" in last_log:
                    prod_pct = 0.40
            st.progress(prod_pct, text=f"Tiến độ sản xuất ({int(prod_pct * 100)}%)...")
            col_cancel, col_txt = st.columns([1.5, 4])
            with col_cancel:
                if st.button("🛑 Dừng Sản Xuất", key="tab2_stop_prod_btn", use_container_width=True):
                    cancel_event = active_run.get("cancel_event")
                    if isinstance(cancel_event, threading.Event):
                        cancel_event.set()
                        st.warning("Đang gửi lệnh dừng quy trình...")
            with col_txt:
                if prod_logs:
                    st.caption(f"Hoạt động gần nhất: `{prod_logs[-1]}`")
            if prod_logs:
                with st.expander("📋 Nhật ký sản xuất thời gian thực", expanded=True):
                    st.code("\n".join(str(l) for l in prod_logs[-60:]), language="text")
    elif isinstance(active_run, dict) and active_run.get("kind") == "production" and active_run.get("status") == "failed":
        st.error(f"Lỗi khi sản xuất Bước 2: {active_run.get('error')}")

    # Filter row
    view_filter = st.radio(
        "Bộ lọc ứng viên",
        [f"Tất cả ({len(candidates)})", f"Đạt chuẩn In Trực tiếp ({direct_count})", f"Hoa văn phẳng ({pattern_count})"],
        horizontal=True,
        key=f"filter_{prefix}",
    )

    displayed_candidates = candidates
    if "Đạt chuẩn In Trực tiếp" in view_filter or "Direct-Print" in view_filter:
        displayed_candidates = [c for c in candidates if c.is_direct_printable]
    elif "Hoa văn phẳng" in view_filter:
        displayed_candidates = [c for c in candidates if "pattern" in c.classification.lower()]

    # Pagination controls
    page_size_options = [12, 24, 48, 96, "Tất cả"]
    p_ctrl1, p_ctrl2 = st.columns([1, 2])
    with p_ctrl1:
        sel_size = st.selectbox(
            "Số mẫu / trang",
            page_size_options,
            index=0,
            key=f"pg_size_{prefix}",
        )
    page_size = len(displayed_candidates) if sel_size == "Tất cả" else int(sel_size)
    page_count = max(1, (len(displayed_candidates) + page_size - 1) // max(1, page_size))
    with p_ctrl2:
        cur_page = st.number_input(
            f"Trang (1 đến {page_count})",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"pg_num_{prefix}",
        )

    start_idx = (int(cur_page) - 1) * page_size
    end_idx = min(start_idx + page_size, len(displayed_candidates))
    page_candidates = displayed_candidates[start_idx:end_idx]

    st.caption(f"Đang hiển thị {len(page_candidates)} / {len(displayed_candidates)} hoa văn (Tổng kho cào: {len(candidates)} mẫu). Đánh dấu checkbox để chọn mẫu in.")

    # Modern Pattern Cards Grid (3 Columns)
    cols = st.columns(3)
    for idx, c in enumerate(page_candidates):
        with cols[idx % 3]:
            with st.container(border=True):
                check_key = f"sel_{prefix}_{c.image_id}"
                st.checkbox(
                    f"Chọn mẫu #{c.image_id[:10]}",
                    key=check_key,
                )
                img_path = Path(c.local_path)
                if not img_path.exists() and run_dir:
                    candidate_rel = run_dir / c.local_path
                    if candidate_rel.exists():
                        img_path = candidate_rel

                if img_path.exists():
                    st.image(str(img_path), width="stretch")
                elif c.image_url:
                    st.image(c.image_url, width="stretch")
                else:
                    st.warning("Không tìm thấy tệp ảnh.")

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**:violet[★ Điểm: {c.image_score:.1f}/100]**")
                with b2:
                    if c.is_direct_printable:
                        st.markdown("**:green[✓ In Trực Tiếp]**")
                    else:
                        st.markdown(f"`{c.classification}`")

                st.caption(f"🏷️ **Trend:** `{c.trend}` | 📐 `{c.width or '?'}x{c.height or '?'} px`")
                if c.reason:
                    st.caption(f"_{c.reason[:100]}..._")
                if c.pin_url:
                    st.link_button("📌 Xem trên Pinterest", c.pin_url)

                # Formatted Clean Markdown Details (NO RAW PYTHON DICTIONARY DUMP!)
                with st.expander("🔍 Chi tiết thẩm định AI & Ảnh gốc"):
                    if img_path.exists():
                        st.image(str(img_path), caption=f"{img_path.name} ({c.width or '?'}x{c.height or '?'} px)", width="stretch")
                    elif c.image_url:
                        st.image(c.image_url, caption=f"Remote URL ({c.width or '?'}x{c.height or '?'} px)", width="stretch")

                    st.markdown(f"""
- **Điểm tổng quan:** `{c.image_score:.1f} / 100`
- **Độ phẳng 2D (Flat Artwork):** `{c.flat_artwork_score * 100:.0f}%`
- **Khả năng in ấn (Printability):** `{c.printability_score * 100:.0f}%`
- **Phân loại ảnh:** `{c.classification}`
- **Đạt chuẩn in trực tiếp:** `{'✓ Có' if c.is_direct_printable else '✗ Cần AI vẽ lại'}`
- **Kích thước gốc:** `{c.width or '?'} x {c.height or '?'} px`
- **Họa tiết nhận diện:** `{c.motifs or 'Không phát hiện'}`
- **Vai trò nguồn:** `{c.source_role or 'pinterest_crawl'}`
- **Đánh giá AI:** _{c.reason or 'Không có ghi chú'}_
""")

    # Bottom Action Bar (for users scrolling down)
    st.divider()
    b_col1, b_col2 = st.columns([2, 1])
    with b_col2:
        st.write("")
    with b_col1:
        bottom_produce_clicked = st.button(
            f"🚀 2. Produce Selected Images ({len(selected_items)} mẫu)",
            type="primary",
            disabled=len(selected_items) == 0 or bool(is_running),
            key=f"bottom_btn_produce_{prefix}",
            use_container_width=True,
        )

    if top_produce_clicked or bottom_produce_clicked:
        if config is None:
            raw_cfg = read_json(run_dir / "config.json")
            if isinstance(raw_cfg, dict):
                config = restore_pipeline_config(raw_cfg, run_dir.parent)
        if config is not None:
            config = replace(
                config,
                design_mode=step2_design_mode,
                task4_ai_limit=max(len(selected_items), config.task4_ai_limit),
            )
            active_run = start_production_run(selected_items, config, run_dir=run_dir)
            st.session_state["trend_product_active_run"] = active_run
            st.rerun()


# -----------------------------------------------------------------------------
# TAB 3: THÀNH PHẨM & MOCKUP AI (DELIVERABLES SHOWCASE)
# -----------------------------------------------------------------------------
def render_deliverables_showcase(
    output_root: Path,
    active_run: dict | None,
    preview_limit: int = 24,
) -> None:
    st.markdown("### 📦 Bước 3: Thành phẩm In ấn & Phối cảnh Mockup AI (Deliverables Showcase)")
    st.caption("Xem trước và tải về toàn bộ file in chất lượng cao 4K CMYK, ảnh phối cảnh phòng khách sống động và phôi cắt sản phẩm.")

    runs = list_run_dirs(output_root)
    if not runs:
        st.info("Chưa có mẻ sản xuất nào trong thư mục output.")
        return

    # Determine which run to showcase
    target_run_name = st.session_state.get("deliverables_selected_run")
    run_by_name = {r.name: r for r in runs}

    if not target_run_name or target_run_name not in run_by_name:
        # Default to active run if finished, otherwise the latest run with prints, or latest run
        if isinstance(active_run, dict) and active_run.get("result"):
            res_dir = active_run["result"].run_dir
            target_run_name = res_dir.name
        else:
            runs_with_prints = [r for r in runs if production_final_files(r, "*.png")]
            target_run_name = runs_with_prints[0].name if runs_with_prints else runs[0].name

    run_options = [r.name for r in runs]
    if "tab3_run_selector" not in st.session_state or st.session_state["tab3_run_selector"] not in run_by_name:
        st.session_state["tab3_run_selector"] = target_run_name

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        chosen_run_name = st.selectbox(
            "Chọn Mẻ Sản Xuất (Run):",
            run_options,
            format_func=lambda x: f"📁 {x} ({format_mtime(run_by_name[x])})",
            key="tab3_run_selector",
        )
    run_dir = run_by_name[chosen_run_name]
    st.session_state["deliverables_selected_run"] = chosen_run_name

    final_pngs = production_final_files(run_dir, "*.png")
    final_jpgs = production_final_files(run_dir, "*.jpg")
    mockups = image_files(run_dir / "mockups")
    ai_backgrounds = ai_background_files(run_dir)
    product_cutouts = image_files(run_dir / "product_cutouts", "*.png")
    product_cutouts_white = image_files(run_dir / "product_cutouts_white", "*.png")
    report_path = run_dir / "report.html"

    # Metrics Showcase Banner
    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Bản in RGB PNG (4K)", len(final_pngs))
        m2.metric("Bản in CMYK JPG (Xưởng)", len(final_jpgs))
        m3.metric("Mockup Phòng khách AI", len(ai_backgrounds))
        m4.metric("Phôi Cắt Sản Phẩm", len(product_cutouts) + len(product_cutouts_white))
        m5.metric("Mockup Phối cảnh", len(mockups))

    if not final_pngs and not ai_backgrounds:
        st.warning(f"Mẻ chạy `{run_dir.name}` chưa có thành phẩm hoàn thiện. Bạn có thể nạp lại mẻ này vào Tab 2 để tuyển chọn và sản xuất.")
        if (run_dir / "candidate_review.json").exists():
            if st.button("🎨 Nạp vào Tab 2 để duyệt mẫu ngay", key=f"tab3_load_t2_{run_dir.name}"):
                rev_data = read_json(run_dir / "candidate_review.json")
                if isinstance(rev_data, dict):
                    st.session_state["active_candidate_package"] = rev_data
                    raw_cfg = read_json(run_dir / "config.json")
                    if isinstance(raw_cfg, dict):
                        st.session_state["active_pipeline_config"] = restore_pipeline_config(raw_cfg, output_root)
                    st.session_state["switch_to_tab"] = 1
                    st.rerun()

    # Deliverables Sub-views (horizontal radio avoids React-Aria nested tabs DOM collision)
    deliv_subview = st.radio(
        "Mục Thành Phẩm:",
        [
            "🖨️ Bản in Siêu Nét 4K (Print Masters)",
            "🛋️ Phối cảnh Phòng khách AI (Lifestyle Mockups)",
            "✂️ Phôi Cắt Sản Phẩm (Cutouts)",
            "🔄 Bảng So Sánh 4 Bước (Compare Matrix)",
            "📄 Báo cáo & Tệp Tin (Report & Files)",
        ],
        horizontal=True,
        key=f"deliv_subview_{run_dir.name}",
    )
    # SUB-TAB 1: PRINT MASTERS (PNG + CMYK JPG WITH DIRECT DOWNLOAD BUTTONS)
    if deliv_subview == "🖨️ Bản in Siêu Nét 4K (Print Masters)":
        st.markdown("#### 🖨️ Bản in Siêu Nét 4K & CMYK Chuẩn Xưởng In")
        st.caption("Mỗi thiết kế xuất xưởng đi kèm cặp file: RGB PNG 4K cho hiển thị trực tuyến & CMYK JPG cho in xưởng.")
        if not final_pngs and not final_jpgs:
            st.info("Chưa có bản in siêu nét nào.")
        else:
            for idx, png_path in enumerate(final_pngs):
                matching_jpg = None
                jpg_candidate_name = png_path.name.replace("_rgb.png", "_cmyk.jpg").replace(".png", ".jpg")
                matching_candidate = png_path.parent / jpg_candidate_name
                if matching_candidate.exists():
                    matching_jpg = matching_candidate
                else:
                    jpg_matches = [j for j in final_jpgs if j.stem.split("_")[0:2] == png_path.stem.split("_")[0:2]]
                    if jpg_matches:
                        matching_jpg = jpg_matches[0]

                png_size_mb = png_path.stat().st_size / (1024 * 1024)
                with st.container(border=True):
                    col_preview, col_actions = st.columns([1, 1.8])
                    with col_preview:
                        st.image(get_image_thumbnail_bytes(str(png_path)), width="stretch")
                    with col_actions:
                        st.markdown(f"### Mẫu Thiết Kế #{idx + 1}: `{png_path.stem}`")
                        st.markdown("**Độ phân giải:** 4000 x 6400 px (hoặc tỷ lệ chuẩn) | **DPI:** 300")
                        st.markdown(f"- **File RGB PNG:** `{png_size_mb:.2f} MB` — Dùng đăng bán Etsy, Shopify, Amazon.")
                        safe_download_button(
                            png_path,
                            f"⬇️ Tải File RGB PNG ({png_size_mb:.1f} MB)",
                            key=f"dl_png_{png_path.name}",
                            mime="image/png",
                        )

                        if matching_jpg and matching_jpg.exists():
                            jpg_size_mb = matching_jpg.stat().st_size / (1024 * 1024)
                            st.markdown(f"- **File CMYK JPG:** `{jpg_size_mb:.2f} MB` — Hệ màu chuẩn in ấn cho xưởng POD.")
                            safe_download_button(
                                matching_jpg,
                                f"⬇️ Tải File CMYK JPG ({jpg_size_mb:.1f} MB)",
                                key=f"dl_jpg_{matching_jpg.name}",
                                mime="image/jpeg",
                            )

    # SUB-TAB 2: LIFESTYLE MOCKUPS
    elif deliv_subview == "🛋️ Phối cảnh Phòng khách AI (Lifestyle Mockups)":
        st.markdown("#### 🛋️ Phối cảnh Phòng khách AI Thực tế (Lifestyle Mockups)")
        st.caption("Gemini render chân thực thảm đặt trong phòng khách hiện đại, chuẩn ánh sáng và nếp gấp tự nhiên.")
        if not ai_backgrounds:
            st.info("Chưa có ảnh phối cảnh phòng khách AI.")
        else:
            cols = st.columns(3)
            for idx, mockup_path in enumerate(ai_backgrounds):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(get_image_thumbnail_bytes(str(mockup_path)), width="stretch")
                        m_size_mb = mockup_path.stat().st_size / (1024 * 1024)
                        st.caption(f"**{mockup_path.name}** ({m_size_mb:.1f} MB)")
                        safe_download_button(
                            mockup_path,
                            f"⬇️ Tải Mockup ({m_size_mb:.1f} MB)",
                            key=f"dl_ai_mockup_{mockup_path.name}_{idx}",
                            mime="image/png",
                        )

    # SUB-TAB 3: PRODUCT CUTOUTS
    elif deliv_subview == "✂️ Phôi Cắt Sản Phẩm (Cutouts)":
        st.markdown("#### ✂️ Phôi Cắt Sản Phẩm (Product Cutouts)")
        st.caption("Ảnh phôi tách nền trong suốt hoặc nền trắng thương mại để ghép mockup tùy chỉnh.")
        cutout_choice = st.radio(
            "Chế độ nền",
            ["Trong suốt (Transparent PNG)", "Nền trắng thương mại (White background)"],
            horizontal=True,
            key=f"{run_dir.name}_cutout_radio",
        )
        selected_cutouts = product_cutouts if "Trong suốt" in cutout_choice else product_cutouts_white
        if not selected_cutouts:
            st.info("Không có phôi cắt tương ứng.")
        else:
            cols = st.columns(3)
            for idx, c_path in enumerate(selected_cutouts):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(get_image_thumbnail_bytes(str(c_path)), width="stretch")
                        st.caption(c_path.name)
                        safe_download_button(
                            c_path,
                            "⬇️ Tải Phôi Cắt",
                            key=f"dl_cutout_{c_path.name}_{idx}",
                            mime="image/png",
                        )

    # SUB-TAB 4: COMPARE MATRIX
    elif deliv_subview == "🔄 Bảng So Sánh 4 Bước (Compare Matrix)":
        st.markdown("#### 🔄 Bảng So Sánh 4 Bước Toàn Diện")
        st.caption("So sánh đối chiếu trực tiếp: Ảnh gốc Pinterest ➔ Phôi cắt tách nền ➔ Bản in chuẩn xưởng ➔ Phối cảnh AI.")
        render_comparison_view(run_dir, preview_limit)

    # SUB-TAB 5: REPORT & FILES
    elif deliv_subview == "📄 Báo cáo & Tệp Tin (Report & Files)":
        st.markdown("#### 📄 Báo Cáo & Danh Mục Tệp Tin Mẻ Chạy")
        st.write("Thư mục lưu trữ trên máy:")
        st.code(str(run_dir))
        if report_path.exists():
            st.markdown("**Báo cáo HTML tổng hợp:**")
            safe_download_button(report_path, "🌐 Tải Báo Cáo report.html", key=f"dl_report_{run_dir.name}", mime="text/html")

        all_final_files = final_pngs + final_jpgs
        if all_final_files:
            st.markdown("**Danh sách file in hoàn thiện:**")
            st.dataframe(
                [
                    {
                        "Tên tệp": path.name,
                        "Kích thước (MB)": round(path.stat().st_size / (1024 * 1024), 2),
                        "Đường dẫn": str(path),
                    }
                    for path in all_final_files
                ],
                width="stretch",
                hide_index=True,
            )


# -----------------------------------------------------------------------------
# TAB 4: KHO LƯU TRỮ & LỊCH SỬ (RUN ARCHIVES)
# -----------------------------------------------------------------------------
def render_run_archives(output_root: Path, preview_limit: int = 24) -> None:
    st.markdown("### 🕒 Bước 4: Kho Lưu trữ & Lịch sử Chạy (Run Archives)")
    st.caption("Quản lý toàn bộ các mẻ cào và mẻ sản xuất trước đây. Nạp lại mẻ cũ vào Tab 2 để tuyển chọn thêm mẫu, hoặc mở Tab 3 để tải lại file in.")

    runs = list_run_dirs(output_root)
    if not runs:
        st.info("Chưa có mẻ chạy nào trong thư mục output.")
        return

    summaries = [run_summary(run_dir) for run_dir in runs]
    run_by_name = {run_dir.name: run_dir for run_dir in runs}

    # Clean run selector
    labels = {
        str(s["run"]): (
            f"📁 {s['run']} | {s['modified']} | Trạng thái: {s['status'].upper()} | "
            f"{s['product']} ({s['size']}) | 4K PNG: {s['final_png']} | AI Mockup: {s['ai_background']} | Ứng viên: {s.get('candidates', 0)}"
        )
        for s in summaries
    }

    if "history_selected_run" not in st.session_state or st.session_state["history_selected_run"] not in run_by_name:
        st.session_state["history_selected_run"] = runs[0].name

    selected_run_name = st.radio(
        "Chọn Mẻ Chạy Cần Xem:",
        options=[r.name for r in runs],
        format_func=lambda name: labels.get(name, name),
        key="history_selected_run",
    )
    run_dir = run_by_name[selected_run_name]

    # Selected run detail container
    final_pngs = production_final_files(run_dir, "*.png")
    final_jpgs = production_final_files(run_dir, "*.jpg")
    mockups = image_files(run_dir / "mockups")
    ai_backgrounds = ai_background_files(run_dir)
    product_cutouts = image_files(run_dir / "product_cutouts", "*.png")
    product_cutouts_white = image_files(run_dir / "product_cutouts_white", "*.png")
    report_path = run_dir / "report.html"
    has_candidate_review = (run_dir / "candidate_review.json").exists()
    cand_review = read_json(run_dir / "candidate_review.json")
    candidates_count = len(cand_review.get("candidates", [])) if isinstance(cand_review, dict) else 0

    with st.container(border=True):
        st.markdown(f"#### 📁 Chi Tiết Mẻ Chạy: `{run_dir.name}`")
        st.caption(f"Đường dẫn: {run_dir}")

        # Prominent 1-Click Action Bar for Tab Switching
        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            if has_candidate_review and candidates_count > 0:
                if st.button(f"🎨 Nạp {candidates_count} mẫu vào Tab 2 để Duyệt lại", key=f"btn_nav_t2_{run_dir.name}", use_container_width=True, type="primary"):
                    st.session_state["active_candidate_package"] = cand_review
                    raw_cfg = read_json(run_dir / "config.json")
                    if isinstance(raw_cfg, dict):
                        st.session_state["active_pipeline_config"] = restore_pipeline_config(raw_cfg, output_root)
                    st.session_state["switch_to_tab"] = 1
                    st.rerun()
            else:
                st.button("🎨 Không có dữ liệu duyệt mẫu", disabled=True, key=f"btn_nav_t2_dis_{run_dir.name}", use_container_width=True)

        with act_col2:
            if final_pngs or ai_backgrounds:
                if st.button("📦 Xem Thành phẩm tại Tab 3", key=f"btn_nav_t3_{run_dir.name}", use_container_width=True, type="primary"):
                    st.session_state["deliverables_selected_run"] = run_dir.name
                    st.session_state["switch_to_tab"] = 2
                    st.rerun()
            else:
                st.button("📦 Chưa có thành phẩm", disabled=True, key=f"btn_nav_t3_dis_{run_dir.name}", use_container_width=True)

        with act_col3:
            if report_path.exists():
                safe_download_button(report_path, "🌐 Tải Báo Cáo HTML", key=f"dl_rep_{run_dir.name}", mime="text/html")
            else:
                st.button("🌐 Chưa có báo cáo", disabled=True, key=f"dl_rep_dis_{run_dir.name}", use_container_width=True)

        # Metrics summary
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Bản in 4K PNG", len(final_pngs))
        m_c2.metric("Bản in CMYK", len(final_jpgs))
        m_c3.metric("Mockup AI", len(ai_backgrounds))
        m_c4.metric("Phôi cắt", len(product_cutouts) + len(product_cutouts_white))
        m_c5.metric("Hoa văn cào", candidates_count)

    # Overview read-only table
    with st.expander("📊 Bảng thống kê toàn bộ các mẻ chạy", expanded=False):
        st.dataframe(summaries, width="stretch", hide_index=True)

    # Config & manifest JSON expander
    with st.expander("🛠️ Cấu hình kỹ thuật & Manifest của mẻ chạy", expanded=False):
        cfg = read_json(run_dir / "config.json")
        stage_m = read_json(run_dir / "stage_manifest.json")
        trend_pkg = read_json(run_dir / "task5_trends" / "trend_package.json")
        st.write("config.json")
        st.json(cfg or {})
        st.write("stage_manifest.json")
        st.json(stage_m or {})
        if trend_pkg:
            st.write("trend_package.json")
            st.json(trend_pkg)

    # Diagnostic reason if crawl failed
    crawl_m = read_json(run_dir / "task5_crawl" / "crawl_manifest.json")
    if not final_pngs and not has_candidate_review:
        render_empty_run_reason(crawl_m, run_dir)


# -----------------------------------------------------------------------------
# MAIN STREAMLIT APP APPLICATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trend Product Tool - POD Studio",
    page_icon="🎨",
    layout="wide",
)

st.title("Trend Product Tool — POD Production Studio")
st.caption("Quy trình tự động hóa POD: Cào xu hướng Pinterest ➔ Chấm điểm AI Vision ➔ Tinh chỉnh & Duyệt hoa văn ➔ Xuất file in 4K CMYK & Mockup AI")

output_root = standalone_output_root()

# Sidebar for global settings & tokens
with st.sidebar:
    st.header("⚙️ Thiết Lập Hệ Thống")

    if st.button("🌐 Mở Chromium đăng nhập Pinterest", help="Mở Chromium để đăng nhập lưu session cho pinterest-browser."):
        try:
            log_p = launch_pinterest_browser_login()
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success("Cửa sổ Pinterest Chromium đã mở. Vui lòng đăng nhập trên trình duyệt rồi quay lại đây.")
            st.caption(f"Tệp log: {log_p}")

    render_pinterest_api_token_ui()

    st.subheader("Cấu hình Đồ họa & AI")
    design_mode = st.selectbox(
        "Chế độ thiết kế mặc định",
        ["direct", "ai_artwork"],
        format_func=lambda x: "Direct Print (In sắc nét hoa văn gốc)" if x == "direct" else "AI Artwork (Gemini vẽ lại)",
        index=0,
        help="Direct Print tăng nét hoa văn gốc Pinterest. AI Artwork nhờ Gemini vẽ lại.",
    )
    artwork_image_size = st.selectbox("Độ phân giải phôi AI", ["1K", "2K", "4K"], index=1)
    gemini_backend = env("GEMINI_BACKEND", "auto")
    gemini_model = env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash")
    trend_region = env("PINTEREST_REGION", "US")
    trend_type = env("PINTEREST_TREND_TYPE", "growing")
    task5_token_path_text = str(task5_token_path_from_env() or "")

    with st.expander("Thư mục Output", expanded=False):
        st.code(str(output_root))

global_settings = {
    "design_mode": design_mode,
    "artwork_image_size": artwork_image_size,
    "gemini_backend": gemini_backend,
    "gemini_model": gemini_model,
    "trend_region": trend_region,
    "trend_type": trend_type,
    "task5_token_path": task5_token_path_text,
}

active_run = st.session_state.get("trend_product_active_run")
is_running = isinstance(active_run, dict) and active_run.get("status") == "running"
run_status = str(active_run.get("status") or "") if isinstance(active_run, dict) else ""

# Handle status transitions and auto-tab switching
if isinstance(active_run, dict):
    if run_status == "review_ready":
        pkg = active_run.get("package")
        if pkg:
            st.session_state["active_candidate_package"] = pkg
        active_run["status"] = "review_ready_handled"
        st.session_state["switch_to_tab"] = 1  # Auto switch to Tab 2
        st.rerun()
    elif run_status == "complete":
        res = active_run.get("result")
        if res is not None:
            st.session_state["deliverables_selected_run"] = res.run_dir.name
        active_run["status"] = "complete_handled"
        st.session_state["switch_to_tab"] = 2  # Auto switch to Tab 3
        st.rerun()

# Check active candidate package (default to latest run if none loaded)
active_pkg = st.session_state.get("active_candidate_package")
if not active_pkg and not is_running:
    latest_runs = list_run_dirs(output_root)
    if latest_runs:
        latest_cand_file = latest_runs[0] / "candidate_review.json"
        if latest_cand_file.exists():
            active_pkg = read_json(latest_cand_file)
            st.session_state["active_candidate_package"] = active_pkg
            raw_c = read_json(latest_runs[0] / "config.json")
            if isinstance(raw_c, dict):
                st.session_state["active_pipeline_config"] = restore_pipeline_config(raw_c, output_root)

# -----------------------------------------------------------------------------
# 4 WORKFLOW TABS
# -----------------------------------------------------------------------------
WORKFLOW_TAB_LABELS = [
    "🔍 1. Khám phá & Cào Trend",
    "🎨 2. Duyệt & Tuyển chọn Hoa văn",
    "📦 3. Thành phẩm & Mockup AI",
    "🕒 4. Kho Lưu trữ & Lịch sử",
]

switch_target = st.session_state.pop("switch_to_tab", None)
if switch_target is not None:
    if isinstance(switch_target, int) and 0 <= switch_target < len(WORKFLOW_TAB_LABELS):
        st.session_state["workflow_active_tab"] = WORKFLOW_TAB_LABELS[switch_target]
    elif switch_target in WORKFLOW_TAB_LABELS:
        st.session_state["workflow_active_tab"] = switch_target

tab_crawl, tab_review, tab_deliverables, tab_history = st.tabs(
    WORKFLOW_TAB_LABELS,
    key="workflow_active_tab",
    on_change="rerun",
)

with tab_crawl:
    render_crawl_tab(output_root, active_run, is_running, global_settings)

with tab_review:
    if active_pkg:
        active_cfg = st.session_state.get("active_pipeline_config")
        render_candidate_review_ui(
            active_pkg,
            config=active_cfg,
            is_running=is_running,
            key_prefix="main_review",
        )
    else:
        st.info("Chưa có danh sách hoa văn nào để duyệt. Vui lòng chạy **Bước 1 (Khám phá & Cào Trend)** hoặc chọn một mẻ cào cũ từ **Tab 4 (Kho Lưu trữ)**.")

with tab_deliverables:
    render_deliverables_showcase(output_root, active_run)

with tab_history:
    render_run_archives(output_root)

# If running, automatically rerun to poll logs and status updates
if is_running:
    time.sleep(0.5)
    st.rerun()
