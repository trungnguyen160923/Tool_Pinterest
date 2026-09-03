#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 - Rug Artwork Replacement Benchmark
Google Cloud / Gemini Enterprise Agent Platform
Supports:
- Standard PayGo
- Flex PayGo
- Folder-based inputs (no hardcoded image paths)
- 1 artwork source + N rug reference images
- Output image generation + cost/latency benchmark
- CSV + HTML report + manual review sheet

Recommended use:
1) Activate your venv
2) Ensure ADC / Google Cloud project is already configured
3) Put:
   - 1 artwork image into an artwork folder
   - many rug reference images into a references folder
4) Run:
   python benchmark_rug_artwork_replacement.py

If you do not pass folders/files via CLI, the script can open file/folder pickers.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Optional GUI picker
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
except Exception as e:
    print("ERROR: google-genai is not installed.")
    print("Install dependencies first. Example:")
    print("  pip install -r requirements_task3.txt")
    raise

# ============================================================
# Config
# ============================================================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROOT_DEFAULT_OUTPUT = "runs_task3_rug_replacement"

MODEL_SPECS = {
    "lite": {
        "label": "Nano Banana 2 Lite",
        "id": "gemini-3.1-flash-lite-image",
        "targets": {"1K"},
    },
    "nb2": {
        "label": "Nano Banana 2",
        "id": "gemini-3.1-flash-image",
        "targets": {"1K", "2K", "4K"},
    },
    "pro": {
        "label": "Nano Banana Pro",
        "id": "gemini-3-pro-image",
        "targets": {"1K", "2K", "4K"},
    },
}

# Pricing on Google Cloud / Agent Platform
# Values are USD / 1,000,000 tokens
PRICING = {
    "standard": {
        "lite": {"input": 0.25, "image_output": 30.0},
        "nb2": {"input": 0.50, "image_output": 60.0},
        "pro": {"input": 2.00, "image_output": 120.0},
    },
    "flex": {
        "lite": {"input": 0.125, "image_output": 15.0},
        "nb2": {"input": 0.25, "image_output": 30.0},
        "pro": {"input": 1.00, "image_output": 60.0},
    }
}

FLEX_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
    "X-Server-Timeout": "1800",
}

STANDARD_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Server-Timeout": "1800",
}

# Retry policy
MAX_ATTEMPTS = 5
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
RETRY_SLEEP_SCHEDULE = {
    "standard": [4, 8, 16, 32],
    "flex": [6, 12, 24, 48],
}

# ============================================================
# Data classes
# ============================================================

@dataclass
class RunConfig:
    project: str
    location: str
    modes: List[str]
    models: List[str]
    targets: List[str]
    artwork_path: str
    reference_dir: str
    output_root: str
    max_refs: Optional[int]
    format: str
    skip_existing: bool
    html_title: str


@dataclass
class ResultRow:
    mode: str
    model_key: str
    model_label: str
    model_id: str
    target: str
    artwork_file: str
    reference_file: str
    reference_width: int
    reference_height: int
    output_file: str
    output_width: Optional[int]
    output_height: Optional[int]
    aspect_ratio: str
    status: str
    traffic_type: str
    attempts_used: int
    retry_count: int
    latency_sec: Optional[float]
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    error_message: str

# ============================================================
# Utilities
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def discover_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    images = [p for p in folder.iterdir() if is_image_file(p)]
    return sorted(images)


def choose_file_dialog(title: str, initial_dir: Optional[str] = None) -> Optional[str]:
    if not TK_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir or os.getcwd(),
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None


def choose_folder_dialog(title: str, initial_dir: Optional[str] = None) -> Optional[str]:
    if not TK_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title, initialdir=initial_dir or os.getcwd())
    root.destroy()
    return path or None


def pick_artwork_file(args) -> Path:
    if args.artwork_file:
        p = Path(args.artwork_file)
        if not p.exists():
            raise FileNotFoundError(f"Artwork file not found: {p}")
        return p

    if args.artwork_dir:
        folder = Path(args.artwork_dir)
        images = discover_images(folder)
        if len(images) == 0:
            raise FileNotFoundError(f"No images found in artwork_dir: {folder}")
        if len(images) == 1:
            return images[0]
        print("\nArtwork folder contains multiple images. Choose one:\n")
        for i, p in enumerate(images, start=1):
            print(f"[{i}] {p.name}")
        while True:
            choice = input("Select artwork file number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                return images[int(choice) - 1]
            print("Invalid choice. Try again.")

    picked = choose_file_dialog("Choose the artwork image")
    if picked:
        return Path(picked)

    raise ValueError("No artwork file selected. Use --artwork-file or --artwork-dir.")


def pick_reference_dir(args) -> Path:
    if args.reference_dir:
        p = Path(args.reference_dir)
        if not p.exists():
            raise FileNotFoundError(f"Reference dir not found: {p}")
        return p

    picked = choose_folder_dialog("Choose the rug reference images folder")
    if picked:
        return Path(picked)

    raise ValueError("No reference_dir selected. Use --reference-dir.")


def pick_output_root(args) -> Path:
    if args.output_root:
        return Path(args.output_root)
    picked = choose_folder_dialog("Choose output root folder (Cancel = current directory)")
    if picked:
        return Path(picked)
    return Path.cwd() / ROOT_DEFAULT_OUTPUT


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def save_image_bytes(img_bytes: bytes, out_path: Path):
    out_path.write_bytes(img_bytes)


def normalize_input_image(img: Image.Image, max_edge: int = 2048) -> Image.Image:
    """
    Keep image visually intact, only downscale if extremely large to reduce request payload size.
    """
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def nearest_aspect_ratio_str(w: int, h: int) -> str:
    target = w / h
    candidates = {
        "1:1": 1 / 1,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
    }
    best = min(candidates.items(), key=lambda kv: abs(kv[1] - target))
    return best[0]


def build_prompt(target: str) -> str:
    return f"""
You are creating a high-quality ecommerce rug image.

Image A is the source Halloween artwork. Use it as the design that must appear on the rug.
Image B is the rug reference photo. Use Image B as the scene, composition, rug shape, rug perspective, lighting, texture, room/background, camera angle, and product presentation reference.

Task:
Create one photorealistic final image by preserving the overall scene of Image B and replacing only the current printed design on the rug with the Halloween artwork from Image A.

Requirements:
- Keep the rug placement, rug boundaries, rug perspective, scale, lighting, shadows, texture, and folds consistent with Image B.
- Preserve the room/background/furniture and overall framing from Image B.
- Apply the Halloween artwork from Image A naturally onto the rug surface.
- Keep the design recognizable and high fidelity: witch silhouette, giant moon, bats, pumpkins, spooky buildings, purple/orange/black Halloween palette.
- Do not add unrelated elements.
- Do not change the rug into another product type.
- Do not significantly alter the scene.
- The result should look like a premium ecommerce/lifestyle rug image.
- Make the design centered and properly fit within the rug area.
- Ensure the output is crisp and clean.

Target output resolution preference: {target}.
Output: image only.
""".strip()


def create_client(project: str, location: str, mode: str):
    headers = FLEX_HEADERS if mode == "flex" else STANDARD_HEADERS
    http_options = types.HttpOptions(
        api_version="v1",
        timeout=1_800_000,
        headers=headers,
    )
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=http_options,
    )
    return client


def make_image_part(img: Image.Image) -> Any:
    img = normalize_input_image(img)
    data = pil_to_jpeg_bytes(img, quality=95)
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")


def extract_usage_metadata(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"prompt_tokens": None, "output_tokens": None, "total_tokens": None}

    def _get(obj, *names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    prompt_tokens = _get(usage, "prompt_token_count", "promptTokenCount")
    output_tokens = _get(usage, "candidates_token_count", "candidatesTokenCount")
    total_tokens = _get(usage, "total_token_count", "totalTokenCount")
    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


def _obj_to_dict(obj) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(exclude_none=True)
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    return {}


def deep_find_first(obj, key_names: set) -> Optional[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in key_names:
                return v
            found = deep_find_first(v, key_names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_first(item, key_names)
            if found is not None:
                return found
    return None


def extract_traffic_type(response) -> str:
    # Prefer direct field if present in response dict
    d = _obj_to_dict(response)
    traffic = deep_find_first(d, {"trafficType", "traffic_type"})
    if traffic is None:
        # Some SDK versions may not expose it. Leave blank.
        return ""
    return str(traffic)


def extract_first_image_bytes(response) -> Tuple[Optional[bytes], str]:
    """
    Works with typical google-genai image responses.
    Returns (bytes, mime_type)
    """
    # First try candidates -> content -> parts
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None:
                data = getattr(inline_data, "data", None)
                mime_type = getattr(inline_data, "mime_type", None) or "image/jpeg"
                if data:
                    return data, mime_type

    # Fallback: traverse dict
    d = _obj_to_dict(response)
    parts = deep_find_first(d, {"parts"})
    if isinstance(parts, list):
        for p in parts:
            inline_data = p.get("inlineData") or p.get("inline_data")
            if inline_data:
                data = inline_data.get("data")
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/jpeg"
                if data:
                    try:
                        return base64.b64decode(data), mime_type
                    except Exception:
                        pass

    return None, ""


def extract_http_code_from_exception(exc: Exception) -> Optional[int]:
    # Try common patterns in google-genai errors
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    s = str(exc)
    for code in [408, 429, 500, 502, 503, 504]:
        if f"{code}" in s:
            return code
    return None


def estimate_cost_usd(mode: str, model_key: str, prompt_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if prompt_tokens is None or output_tokens is None:
        return None
    price = PRICING[mode][model_key]
    in_cost = (prompt_tokens / 1_000_000) * price["input"]
    out_cost = (output_tokens / 1_000_000) * price["image_output"]
    return in_cost + out_cost


def score_value_simple(cost: Optional[float], latency: Optional[float]) -> Optional[float]:
    """
    Small helper for ranking only.
    Lower cost and lower latency => higher score.
    Quality is NOT auto-scored here.
    """
    if cost is None or latency is None:
        return None
    return 1.0 / max(1e-9, (cost * 0.7 + latency * 0.3 / 1000.0))


def generate_one(
    client,
    mode: str,
    model_key: str,
    target: str,
    artwork_img: Image.Image,
    reference_img: Image.Image,
    aspect_ratio: str,
) -> Tuple[Optional[bytes], Dict[str, Any], Optional[Exception], int, float]:
    model_id = MODEL_SPECS[model_key]["id"]
    prompt = build_prompt(target)
    last_exc = None
    total_start = time.perf_counter()

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.2,
    )

    if hasattr(config, "image_config"):
        # some SDK versions may accept image_config
        try:
            config.image_config = types.ImageConfig(aspect_ratio=aspect_ratio)
        except Exception:
            pass

    contents = [
        prompt,
        make_image_part(artwork_img),
        make_image_part(reference_img),
    ]

    schedule = RETRY_SLEEP_SCHEDULE[mode]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            latency_sec = time.perf_counter() - start
            img_bytes, mime_type = extract_first_image_bytes(response)
            usage = extract_usage_metadata(response)
            traffic_type = extract_traffic_type(response)

            if img_bytes is None:
                raise RuntimeError("No image bytes found in response.")

            meta = {
                "mime_type": mime_type or "image/jpeg",
                "usage": usage,
                "traffic_type": traffic_type,
                "response_dict": _obj_to_dict(response),
            }
            return img_bytes, meta, None, attempt, latency_sec
        except Exception as e:
            last_exc = e
            code = extract_http_code_from_exception(e)
            if attempt >= MAX_ATTEMPTS or code not in RETRYABLE_HTTP_CODES:
                break
            wait_sec = schedule[min(attempt - 1, len(schedule) - 1)]
            print(f"    retryable error (mode={mode}, model={model_key}, attempt={attempt}/{MAX_ATTEMPTS}, code={code})")
            print(f"    waiting {wait_sec}s ...")
            time.sleep(wait_sec)

    total_elapsed = time.perf_counter() - total_start
    return None, {"usage": {}, "traffic_type": "", "total_elapsed": total_elapsed}, last_exc, MAX_ATTEMPTS, total_elapsed


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def make_thumbnail(src: Path, dst: Path, max_side: int = 320):
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(dst, format="JPEG", quality=88)


def format_money(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"${v:,.6f}"


def format_float(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{digits}f}"


def generate_html_report(run_dir: Path, results_df: pd.DataFrame, summary_df: pd.DataFrame, config: RunConfig):
    html_path = run_dir / "report.html"
    thumbs_dir = safe_mkdir(run_dir / "thumbnails")

    # create thumbs
    for _, row in results_df.iterrows():
        out_file = row.get("output_file", "")
        if out_file and isinstance(out_file, str):
            p = run_dir / out_file
            if p.exists():
                thumb = thumbs_dir / (Path(out_file).stem + "_thumb.jpg")
                try:
                    make_thumbnail(p, thumb)
                except Exception:
                    pass

    rows_html = []
    for _, row in results_df.iterrows():
        out_rel = row.get("output_file", "")
        thumb_rel = ""
        if out_rel:
            thumb_path = thumbs_dir / (Path(out_rel).stem + "_thumb.jpg")
            if thumb_path.exists():
                thumb_rel = relative_to(thumb_path, run_dir)

        img_html = f'<a href="{out_rel}"><img src="{thumb_rel}" style="max-width:180px;border:1px solid #ccc;"></a>' if thumb_rel else ""
        rows_html.append(f"""
        <tr>
            <td>{row.get('mode','')}</td>
            <td>{row.get('model_label','')}</td>
            <td>{row.get('target','')}</td>
            <td>{row.get('reference_file','')}</td>
            <td>{row.get('status','')}</td>
            <td>{format_money(row.get('estimated_cost_usd'))}</td>
            <td>{format_float(row.get('latency_sec'),2)}</td>
            <td>{row.get('traffic_type','')}</td>
            <td>{row.get('attempts_used','')}</td>
            <td>{row.get('prompt_tokens','')}</td>
            <td>{row.get('output_tokens','')}</td>
            <td>{img_html}</td>
            <td style="max-width:350px;word-break:break-word;">{row.get('error_message','')}</td>
        </tr>
        """)

    summary_rows = []
    for _, row in summary_df.iterrows():
        summary_rows.append(f"""
        <tr>
            <td>{row.get('mode','')}</td>
            <td>{row.get('model_label','')}</td>
            <td>{row.get('target','')}</td>
            <td>{row.get('num_success','')}</td>
            <td>{format_money(row.get('avg_cost_usd'))}</td>
            <td>{format_float(row.get('avg_latency_sec'),2)}</td>
            <td>{format_float(row.get('value_score'),6)}</td>
            <td>{row.get('suggestion','')}</td>
        </tr>
        """)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{config.html_title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1, h2, h3 {{ color: #17365D; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #17365D; color: white; }}
.note {{ background: #FFF2CC; border: 1px solid #E0C97F; padding: 12px; margin: 16px 0; }}
.small {{ color: #666; font-size: 13px; }}
code {{ background: #f5f5f5; padding: 2px 5px; }}
</style>
</head>
<body>
<h1>{config.html_title}</h1>

<div class="note">
<b>Task:</b> 1 source Halloween artwork + N rug reference images → generate rug images with the artwork applied onto each rug.<br>
<b>Backend:</b> Google Cloud / Gemini Enterprise Agent Platform<br>
<b>Modes tested:</b> {", ".join(config.modes)}<br>
<b>Models tested:</b> {", ".join(config.models)}<br>
<b>Targets:</b> {", ".join(config.targets)}<br>
<b>Artwork:</b> {Path(config.artwork_path).name}<br>
<b>Reference folder:</b> {config.reference_dir}
</div>

<h2>Summary</h2>
<table>
<thead>
<tr>
<th>Mode</th>
<th>Model</th>
<th>Target</th>
<th>Success count</th>
<th>Avg cost / image</th>
<th>Avg latency</th>
<th>Value score</th>
<th>Suggestion</th>
</tr>
</thead>
<tbody>
{''.join(summary_rows)}
</tbody>
</table>

<h2>Detailed Results</h2>
<table>
<thead>
<tr>
<th>Mode</th>
<th>Model</th>
<th>Target</th>
<th>Reference</th>
<th>Status</th>
<th>Estimated cost</th>
<th>Latency (s)</th>
<th>Traffic type</th>
<th>Attempts</th>
<th>Prompt tokens</th>
<th>Output tokens</th>
<th>Output preview</th>
<th>Error</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>

<h2>How to interpret this report</h2>
<div class="small">
<ul>
<li><b>Cost</b> is estimated from response usage metadata and current Google Cloud / Agent Platform pricing.</li>
<li><b>Value score</b> in this report is a rough ranking from cost + latency only.</li>
<li><b>Important:</b> quality for this task should still be reviewed visually. The best final choice depends on:
    <ul>
        <li>design fidelity (does the Halloween artwork stay correct?)</li>
        <li>scene preservation (does the rug photo scene stay intact?)</li>
        <li>product realism (does the design sit naturally on the rug?)</li>
    </ul>
</li>
<li>Use <code>manual_review_template.csv</code> to score visual quality.</li>
</ul>
</div>

</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def create_manual_review_template(run_dir: Path, results_df: pd.DataFrame):
    rows = []
    for _, row in results_df.iterrows():
        if row.get("status") != "success":
            continue
        rows.append({
            "mode": row["mode"],
            "model_key": row["model_key"],
            "model_label": row["model_label"],
            "target": row["target"],
            "reference_file": row["reference_file"],
            "output_file": row["output_file"],
            "design_fidelity_1to5": "",
            "scene_preservation_1to5": "",
            "product_realism_1to5": "",
            "overall_1to5": "",
            "notes": "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "manual_review_template.csv", index=False, encoding="utf-8-sig")


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    ok = results_df[results_df["status"] == "success"].copy()
    if ok.empty:
        return pd.DataFrame(columns=[
            "mode", "model_key", "model_label", "target",
            "num_success", "avg_cost_usd", "avg_latency_sec", "value_score", "suggestion"
        ])

    agg = (
        ok.groupby(["mode", "model_key", "model_label", "target"], as_index=False)
        .agg(
            num_success=("status", "count"),
            avg_cost_usd=("estimated_cost_usd", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
        )
    )
    agg["value_score"] = agg.apply(lambda r: score_value_simple(r["avg_cost_usd"], r["avg_latency_sec"]), axis=1)

    # suggestion per target & mode = highest value score
    suggestions = []
    for (mode, target), group in agg.groupby(["mode", "target"]):
        group_sorted = group.sort_values(["value_score"], ascending=False)
        winner = group_sorted.iloc[0]
        for i in range(len(group_sorted)):
            row = group_sorted.iloc[i].to_dict()
            row["suggestion"] = "recommended" if i == 0 else ""
            suggestions.append(row)

    final = pd.DataFrame(suggestions)
    return final


def print_run_plan(config: RunConfig, ref_paths: List[Path]):
    print("=" * 72)
    print("TASK 3 - RUG ARTWORK REPLACEMENT BENCHMARK")
    print("=" * 72)
    print(f"Project        : {config.project}")
    print(f"Location       : {config.location}")
    print(f"Modes          : {', '.join(config.modes)}")
    print(f"Models         : {', '.join(config.models)}")
    print(f"Targets        : {', '.join(config.targets)}")
    print(f"Artwork file   : {config.artwork_path}")
    print(f"Reference dir  : {config.reference_dir}")
    print(f"Reference imgs : {len(ref_paths)}")
    print(f"Output root    : {config.output_root}")
    print("-" * 72)

    total_requests = 0
    for model_key in config.models:
        valid_targets = [t for t in config.targets if t in MODEL_SPECS[model_key]["targets"]]
        total_requests += len(config.modes) * len(valid_targets) * len(ref_paths)
    print(f"Total planned requests: {total_requests}")
    print("=" * 72)


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark rug artwork replacement on Google Cloud / Gemini Agent Platform.")
    parser.add_argument("--artwork-file", type=str, default="", help="Path to the 1 artwork image.")
    parser.add_argument("--artwork-dir", type=str, default="", help="Folder containing the artwork image. If multiple images exist, you can choose one.")
    parser.add_argument("--reference-dir", type=str, default="", help="Folder containing rug reference images.")
    parser.add_argument("--output-root", type=str, default="", help="Root output folder.")
    parser.add_argument("--project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT", ""), help="Google Cloud project.")
    parser.add_argument("--location", type=str, default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), help="Google Cloud location, normally global.")
    parser.add_argument("--modes", nargs="+", default=["standard", "flex"], choices=["standard", "flex"], help="Consumption modes to test.")
    parser.add_argument("--models", nargs="+", default=["lite", "nb2", "pro"], choices=["lite", "nb2", "pro"], help="Models to test.")
    parser.add_argument("--targets", nargs="+", default=["1K"], choices=["1K", "2K", "4K"], help="Target output resolution preference.")
    parser.add_argument("--max-refs", type=int, default=0, help="Max reference images to process. 0 = all.")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "jpeg", "png"], help="Saved output image format.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Only print run plan and exit.")
    parser.add_argument("--html-title", type=str, default="Task 3 - Rug Artwork Replacement Benchmark", help="HTML report title.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.project:
        print("ERROR: GOOGLE_CLOUD_PROJECT is empty.")
        print("Set it in env or pass --project.")
        sys.exit(1)

    artwork_path = pick_artwork_file(args)
    reference_dir = pick_reference_dir(args)
    output_root = pick_output_root(args)

    ref_paths = discover_images(reference_dir)
    if not ref_paths:
        print(f"ERROR: no reference images found in {reference_dir}")
        sys.exit(1)

    max_refs = None if not args.max_refs or args.max_refs <= 0 else args.max_refs
    if max_refs is not None:
        ref_paths = ref_paths[:max_refs]

    config = RunConfig(
        project=args.project,
        location=args.location,
        modes=args.modes,
        models=args.models,
        targets=args.targets,
        artwork_path=str(artwork_path),
        reference_dir=str(reference_dir),
        output_root=str(output_root),
        max_refs=max_refs,
        format=args.format,
        skip_existing=args.skip_existing,
        html_title=args.html_title,
    )

    print_run_plan(config, ref_paths)
    if args.dry_run:
        return

    if not ask_yes_no("Proceed with generation?", default=False):
        print("Cancelled.")
        return

    run_dir = safe_mkdir(Path(config.output_root) / now_str())
    outputs_dir = safe_mkdir(run_dir / "outputs")
    logs_dir = safe_mkdir(run_dir / "logs")

    # save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config": asdict(config),
        "model_specs": MODEL_SPECS,
        "pricing": PRICING,
        "retry_policy": {
            "max_attempts": MAX_ATTEMPTS,
            "retryable_http_codes": sorted(RETRYABLE_HTTP_CODES),
            "retry_sleep_schedule": RETRY_SLEEP_SCHEDULE,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    clients = {}
    for mode in config.modes:
        clients[mode] = create_client(config.project, config.location, mode)

    artwork_img = load_image(artwork_path)
    aw_name = artwork_path.name

    results: List[ResultRow] = []

    for ref_idx, ref_path in enumerate(ref_paths, start=1):
        ref_img = load_image(ref_path)
        ref_w, ref_h = ref_img.size
        aspect_ratio = nearest_aspect_ratio_str(ref_w, ref_h)

        print("\n" + "=" * 72)
        print(f"REFERENCE {ref_idx}/{len(ref_paths)}: {ref_path.name} ({ref_w}x{ref_h}, aspect ~ {aspect_ratio})")
        print("=" * 72)

        for mode in config.modes:
            for model_key in config.models:
                valid_targets = [t for t in config.targets if t in MODEL_SPECS[model_key]["targets"]]
                if not valid_targets:
                    continue

                for target in valid_targets:
                    model_label = MODEL_SPECS[model_key]["label"]
                    model_id = MODEL_SPECS[model_key]["id"]

                    out_name = f"{mode}__{model_key}__{target}__{ref_path.stem}.{config.format}"
                    out_path = outputs_dir / out_name
                    output_rel = relative_to(out_path, run_dir)

                    if config.skip_existing and out_path.exists():
                        out_w, out_h = image_size(out_path)
                        results.append(ResultRow(
                            mode=mode,
                            model_key=model_key,
                            model_label=model_label,
                            model_id=model_id,
                            target=target,
                            artwork_file=aw_name,
                            reference_file=ref_path.name,
                            reference_width=ref_w,
                            reference_height=ref_h,
                            output_file=output_rel,
                            output_width=out_w,
                            output_height=out_h,
                            aspect_ratio=aspect_ratio,
                            status="skipped_existing",
                            traffic_type="",
                            attempts_used=0,
                            retry_count=0,
                            latency_sec=None,
                            prompt_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            estimated_cost_usd=None,
                            error_message="",
                        ))
                        print(f"SKIP existing: {out_name}")
                        continue

                    print(f"{mode.upper():8} | {model_label:20} | {target:2} | {ref_path.name}")

                    client = clients[mode]
                    err_msg = ""
                    status = "success"
                    out_w = None
                    out_h = None

                    try:
                        img_bytes, meta, err, attempts_used, latency_sec = generate_one(
                            client=client,
                            mode=mode,
                            model_key=model_key,
                            target=target,
                            artwork_img=artwork_img,
                            reference_img=ref_img,
                            aspect_ratio=aspect_ratio,
                        )

                        if err is not None or img_bytes is None:
                            raise err or RuntimeError("Generation failed: no output image.")

                        save_image_bytes(img_bytes, out_path)
                        out_w, out_h = image_size(out_path)

                        usage = meta.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        output_tokens = usage.get("output_tokens")
                        total_tokens = usage.get("total_tokens")
                        traffic_type = meta.get("traffic_type", "")

                        est_cost = estimate_cost_usd(mode, model_key, prompt_tokens, output_tokens)
                        retry_count = max(0, attempts_used - 1)

                        print(f"  OK -> {out_path.name} | {out_w}x{out_h} | cost={format_money(est_cost)} | latency={format_float(latency_sec)}s | attempts={attempts_used}")

                        results.append(ResultRow(
                            mode=mode,
                            model_key=model_key,
                            model_label=model_label,
                            model_id=model_id,
                            target=target,
                            artwork_file=aw_name,
                            reference_file=ref_path.name,
                            reference_width=ref_w,
                            reference_height=ref_h,
                            output_file=output_rel,
                            output_width=out_w,
                            output_height=out_h,
                            aspect_ratio=aspect_ratio,
                            status=status,
                            traffic_type=traffic_type,
                            attempts_used=attempts_used,
                            retry_count=retry_count,
                            latency_sec=latency_sec,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            estimated_cost_usd=est_cost,
                            error_message="",
                        ))
                    except Exception as e:
                        status = "error"
                        err_msg = f"{type(e).__name__}: {e}"
                        print(f"  ERROR: {err_msg}")

                        tb_path = logs_dir / f"error__{mode}__{model_key}__{target}__{ref_path.stem}.txt"
                        tb_path.write_text(traceback.format_exc(), encoding="utf-8")

                        results.append(ResultRow(
                            mode=mode,
                            model_key=model_key,
                            model_label=model_label,
                            model_id=model_id,
                            target=target,
                            artwork_file=aw_name,
                            reference_file=ref_path.name,
                            reference_width=ref_w,
                            reference_height=ref_h,
                            output_file=output_rel if out_path.exists() else "",
                            output_width=out_w,
                            output_height=out_h,
                            aspect_ratio=aspect_ratio,
                            status=status,
                            traffic_type="",
                            attempts_used=0,
                            retry_count=0,
                            latency_sec=None,
                            prompt_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            estimated_cost_usd=None,
                            error_message=err_msg,
                        ))

    results_df = pd.DataFrame([asdict(r) for r in results])
    results_csv = run_dir / "results.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    summary_df = build_summary(results_df)
    summary_csv = run_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    create_manual_review_template(run_dir, results_df)
    generate_html_report(run_dir, results_df, summary_df, config)

    # Console summary
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Run folder    : {run_dir}")
    print(f"Results CSV   : {results_csv}")
    print(f"Summary CSV   : {summary_csv}")
    print(f"HTML report   : {run_dir / 'report.html'}")
    print(f"Manual review : {run_dir / 'manual_review_template.csv'}")

    if not summary_df.empty:
        print("\nTop recommendations by mode/target:")
        display_cols = ["mode", "target", "model_label", "avg_cost_usd", "avg_latency_sec", "suggestion"]
        show_df = summary_df[summary_df["suggestion"] == "recommended"][display_cols].copy()
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
            print(show_df.to_string(index=False))
    else:
        print("\nNo successful generations yet, so summary is empty.")


if __name__ == "__main__":
    main()
