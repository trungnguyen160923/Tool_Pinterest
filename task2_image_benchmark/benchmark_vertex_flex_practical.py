#!/usr/bin/env python3
"""
Gemini Image Upscale Benchmark — Flex PayGo (Google Cloud / Agent Platform)

Purpose
-------
Benchmark image enhancement/upscaling quality, latency, and estimated API cost
across:
  - Gemini 3.1 Flash-Lite Image (Nano Banana 2 Lite)
  - Gemini 3.1 Flash Image (Nano Banana 2)
  - Gemini 3 Pro Image (Nano Banana Pro)

Benchmark modes
---------------
Default: practical

practical:
  The same normalized source image is sent to every model/target:
    source -> 1K
    source -> 2K
    source -> 4K
  This matches the real production question: given one ordinary image, which
  model preserves it best while increasing resolution, and what does each image cost?

scientific:
  Synthetic low-resolution inputs are created from the source:
    512px input  -> 1K output
    1024px input -> 2K output
    1024px input -> 4K output
  If the source is genuinely high resolution, the original can serve as native
  high-resolution ground truth for SSIM/PSNR/MAE.

Important
---------
Gemini image models are generative image editors, not deterministic classical
upscalers. Objective pixel metrics are diagnostics, not the sole quality score.
Always review fidelity, hallucination, text/logo accuracy, and color manually.

Flex PayGo note
---------------
This version explicitly routes every generateContent request to Flex PayGo using
the Google Cloud request header X-Vertex-AI-LLM-Shared-Request-Type: flex.
Flex is synchronous but latency-tolerant, can be slower, and can experience higher
throttling than Standard. The script verifies usage_metadata.traffic_type is
ON_DEMAND_FLEX so Flex pricing is never silently applied to Standard traffic.

Authentication
--------------
Uses Google Cloud Application Default Credentials (ADC). No Gemini API key.

Recommended setup:
  gcloud auth application-default login
  gcloud auth application-default set-quota-project YOUR_PROJECT_ID

.env:
  GOOGLE_CLOUD_PROJECT=gemini-image-benchmark
  GOOGLE_CLOUD_LOCATION=global
  GOOGLE_GENAI_USE_ENTERPRISE=True
  # Optional, only for VND display:
  # USD_TO_VND=26000

Pricing snapshot
----------------
Pricing below is a snapshot for the global Flex PayGo endpoint and is used
only to estimate request cost. Google Cloud Billing is the authoritative source
for actual billed spend, credits, discounts, taxes, and rounding.

Snapshot date: 2026-08-26
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import fnmatch
import random
import json
import math
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from google import genai
from google.genai import errors, types


# ============================================================
# PATHS / ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RUNS_DIR = BASE_DIR / "runs_flex"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

PROJECT_ID = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
LOCATION = (os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip()
USE_ENTERPRISE = (os.getenv("GOOGLE_GENAI_USE_ENTERPRISE") or "").strip().lower()

try:
    USD_TO_VND = float((os.getenv("USD_TO_VND") or "0").strip())
except ValueError:
    USD_TO_VND = 0.0

if not PROJECT_ID:
    raise RuntimeError(
        "Thiếu GOOGLE_CLOUD_PROJECT trong .env.\n"
        "Ví dụ: GOOGLE_CLOUD_PROJECT=gemini-image-benchmark"
    )

if USE_ENTERPRISE not in {"true", "1", "yes"}:
    raise RuntimeError(
        "GOOGLE_GENAI_USE_ENTERPRISE phải = True trong .env."
    )

if LOCATION.lower() != "global":
    raise RuntimeError(
        "Flex PayGo cho các model benchmark này chỉ hỗ trợ endpoint global.\n"
        "Hãy đặt GOOGLE_CLOUD_LOCATION=global trong .env."
    )


# ============================================================
# BENCHMARK CONFIG
# ============================================================

PRICING_SNAPSHOT_DATE = "2026-08-26"
OUTPUT_MIME_TYPE = "image/jpeg"
OUTPUT_JPEG_QUALITY = 100

# Flex is intentionally latency-tolerant and can throttle more aggressively.
# Keep only a small pacing delay between completed benchmark cases.
DEFAULT_REQUEST_DELAY_SECONDS = 1.0

# Do not add artificial 4K pre-wait in Flex: we want measured turnaround to
# represent the actual Flex request + retry behavior.
DEFAULT_NB2_4K_PRE_REQUEST_DELAY_SECONDS = 0.0

# Official Flex docs allow request timeout up to 30 minutes.
FLEX_SERVER_TIMEOUT_SECONDS = 1800
FLEX_TRAFFIC_TYPE_REQUIRED = "ON_DEMAND_FLEX"
FLEX_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
    "X-Server-Timeout": str(FLEX_SERVER_TIMEOUT_SECONDS),
}

# We implement retries explicitly instead of stacking our own retry loop on top
# of SDK retries. This keeps the number of attempts predictable and lets the
# report record retry count/wait time accurately.
DEFAULT_MAX_ATTEMPTS = 6
RETRY_429_SCHEDULE_SECONDS = (10.0, 20.0, 40.0, 80.0, 120.0)
RETRY_5XX_SCHEDULE_SECONDS = (2.0, 4.0, 8.0, 16.0, 30.0)
RETRY_JITTER_RATIO = 0.20
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}

# Keep benchmark generation as deterministic as the model/API allows.
TEMPERATURE = 0.0

# Inline image requests have a service size limit. Prepared inputs are only
# 512/1024 px by default, but we still validate before sending.
INLINE_IMAGE_MAX_BYTES = 7 * 1024 * 1024

# Synthetic benchmark:
#   512 long-edge  -> 1K
#   1024 long-edge -> 2K
#   1024 long-edge -> 4K
INPUT_LONG_EDGE_BY_TARGET = {
    "1K": 512,
    "2K": 1024,
    "4K": 1024,
}

# Used for preflight warnings only. The actual output dimensions are always
# read from the bytes returned by the model.
NOMINAL_LONG_EDGE_BY_TARGET = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}

PROMPT = """
Enhance and upscale the provided image while preserving the source image
as faithfully as possible.

Preserve the exact framing, composition, object geometry, proportions, people,
facial identity, colors, patterns, artwork, text, logos, materials, textures,
lighting, shadows, background, depth of field, and all existing visual elements.

Improve perceived sharpness and restore fine details only when those details are
genuinely supported by the source image.

Reduce blur, aliasing, noise, JPEG artifacts, and compression artifacts.

Do not add, remove, redesign, reinterpret, beautify, replace, move, crop, extend,
or modify any object or visual element.

Do not invent details that are not supported by the source.

The output must remain photorealistic and highly faithful to the input image.
""".strip()


# ============================================================
# MODEL / PRICING DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    model_id: str
    sizes: tuple[str, ...]
    input_per_1m: float
    text_thinking_output_per_1m: float
    image_output_per_1m: float
    expected_image_tokens: dict[str, int]
    aspect_ratios: frozenset[str]


COMMON_RATIOS = frozenset({
    "1:1",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
})

RATIOS_WITH_9_21 = frozenset(set(COMMON_RATIOS) | {"9:21"})

MODELS: dict[str, ModelConfig] = {
    "lite": ModelConfig(
        key="lite",
        name="Nano Banana 2 Lite",
        model_id="gemini-3.1-flash-lite-image",
        sizes=("1K",),
        input_per_1m=0.125,
        text_thinking_output_per_1m=0.75,
        image_output_per_1m=15.00,
        expected_image_tokens={
            "1K": 1120,
        },
        aspect_ratios=COMMON_RATIOS,
    ),
    "nb2": ModelConfig(
        key="nb2",
        name="Nano Banana 2",
        model_id="gemini-3.1-flash-image",
        sizes=("1K", "2K", "4K"),
        input_per_1m=0.25,
        text_thinking_output_per_1m=1.50,
        image_output_per_1m=30.00,
        expected_image_tokens={
            "1K": 1120,
            "2K": 1680,
            "4K": 2520,
        },
        aspect_ratios=RATIOS_WITH_9_21,
    ),
    "pro": ModelConfig(
        key="pro",
        name="Nano Banana Pro",
        model_id="gemini-3-pro-image",
        sizes=("1K", "2K", "4K"),
        input_per_1m=1.00,
        text_thinking_output_per_1m=6.00,
        image_output_per_1m=60.00,
        expected_image_tokens={
            "1K": 1120,
            "2K": 1120,
            "4K": 2000,
        },
        aspect_ratios=RATIOS_WITH_9_21,
    ),
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemini image enhancement/upscaling using Google Cloud Flex PayGo."
    )
    parser.add_argument(
        "--mode",
        choices=["practical", "scientific"],
        default="practical",
        help=(
            "Benchmark mode. practical = send the same normalized source image "
            "to every model/target (real-world production test). "
            "scientific = synthesize 512/1024px low-res inputs from a higher-res "
            "source so native ground truth can be measured. Default: practical."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Số lần lặp mỗi model/target. Khuyến nghị 3 khi benchmark chính thức.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Model cần chạy. Mặc định: lite nb2 pro",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["1K", "2K", "4K"],
        default=["1K", "2K", "4K"],
        help="Target cần chạy. Model không hỗ trợ target sẽ tự bỏ qua.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Không hỏi xác nhận trước khi phát sinh chi phí.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ preflight + cost preview; không gọi API.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Không tính SSIM/PSNR/Sharpness/MAE.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "Chỉ chạy một số source theo filename/glob. Ví dụ: "
            "--sources img.jpg img3.jpg 'product-*.jpg'"
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help=(
            "Khoảng nghỉ giữa các test case hoàn tất. "
            f"Mặc định {DEFAULT_REQUEST_DELAY_SECONDS}s."
        ),
    )
    parser.add_argument(
        "--nb2-4k-pre-delay",
        type=float,
        default=DEFAULT_NB2_4K_PRE_REQUEST_DELAY_SECONDS,
        help=(
            "Optional pre-delay trước Nano Banana 2 4K (Flex default = 0). "
            f"Mặc định {DEFAULT_NB2_4K_PRE_REQUEST_DELAY_SECONDS}s."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "Tổng số attempt tối đa cho 408/429/5xx. "
            f"Mặc định {DEFAULT_MAX_ATTEMPTS}."
        ),
    )
    return parser.parse_args()


# ============================================================
# CLIENT
# ============================================================

def build_client() -> genai.Client:
    """
    Build an explicit Google Cloud client using ADC.

    Important: SDK-level retries are intentionally NOT configured here. The
    benchmark owns a single retry layer so attempts/backoff are deterministic
    and measurable in results.csv/report.html.
    """
    http_options = types.HttpOptions(
        api_version="v1",
        # Flex is a synchronous latency-tolerant tier. Official docs allow up to
        # 30 minutes. HttpOptions.timeout is milliseconds.
        timeout=FLEX_SERVER_TIMEOUT_SECONDS * 1000,
        # IMPORTANT: Agent Platform/Vertex uses request HEADERS to select Flex.
        # Do not use GenerateContentConfig.service_tier here.
        headers=FLEX_HEADERS,
        # attempts=1 disables SDK retries so the benchmark has exactly one
        # measurable retry layer in generate().
        retry_options=types.HttpRetryOptions(
            attempts=1,
            http_status_codes=sorted(RETRYABLE_HTTP_CODES),
        ),
    )

    return genai.Client(
        enterprise=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=http_options,
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(value: int | float) -> str:
    value = float(value)
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def money_usd(value: float) -> str:
    return f"${value:.6f}"


def money_vnd(value_usd: float) -> str:
    if USD_TO_VND <= 0:
        return ""
    return f"{value_usd * USD_TO_VND:,.0f} VND"


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_num(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "N/A"
    if math.isinf(number):
        return "∞"
    return f"{number:.{digits}f}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mime_for_path(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime in {"image/jpeg", "image/png", "image/webp"}:
        return mime
    # Prepared benchmark inputs are PNG unless fallback conversion is used.
    return "image/png"


def extension_for_mime(mime: str | None) -> str:
    mime = (mime or "").lower()
    if mime == "image/jpeg":
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".bin"


def discover_sources() -> list[Path]:
    extensions = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
        "*.bmp",
        "*.tif",
        "*.tiff",
    )
    sources: list[Path] = []
    for ext in extensions:
        sources.extend(DATASET_DIR.glob(ext))
    return sorted(set(sources), key=lambda p: p.name.lower())


def filter_sources(
    sources: list[Path],
    patterns: list[str] | None,
) -> list[Path]:
    """
    Filter discovered sources by exact filename or shell-style glob.

    This is useful for cheap retries, e.g. rerunning only the four images whose
    NB2 4K requests failed instead of paying for the whole matrix again.
    """
    if not patterns:
        return sources

    selected: list[Path] = []

    for source in sources:
        if any(
            source.name == pattern
            or fnmatch.fnmatch(source.name, pattern)
            for pattern in patterns
        ):
            selected.append(source)

    return selected


# ============================================================
# IMAGE PREPARATION
# ============================================================

def open_rgb(path: Path) -> Image.Image:
    """
    Apply EXIF orientation before converting to RGB.
    This avoids benchmarking a rotated image accidentally.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def classify_resolution(width: int, height: int) -> str:
    long_edge = max(width, height)
    if long_edge >= 6144:
        return "6K+"
    if long_edge >= 4096:
        return "4K+"
    if long_edge >= 3840:
        return "Near 4K"
    if long_edge >= 2048:
        return "2K+"
    if long_edge >= 1024:
        return "1K+"
    return "Below 1K"


def parse_ratio(label: str) -> float:
    a, b = label.split(":")
    return float(a) / float(b)


def ratio_intersection(model_keys: Iterable[str]) -> frozenset[str]:
    sets = [MODELS[key].aspect_ratios for key in model_keys]
    if not sets:
        raise ValueError("Không có model nào được chọn.")
    result = set(sets[0])
    for s in sets[1:]:
        result.intersection_update(s)
    if not result:
        raise RuntimeError("Các model đã chọn không có aspect ratio chung.")
    return frozenset(result)


def closest_supported_aspect_ratio(
    width: int,
    height: int,
    allowed_ratios: Iterable[str],
) -> tuple[str, float]:
    """
    Log-distance treats portrait/landscape ratio errors symmetrically.
    """
    source_ratio = width / height
    allowed = list(allowed_ratios)

    label = min(
        allowed,
        key=lambda key: abs(math.log(source_ratio / parse_ratio(key))),
    )
    return label, parse_ratio(label)


def center_crop_to_ratio(
    image: Image.Image,
    target_ratio: float,
) -> tuple[Image.Image, float]:
    """
    Center-crop without stretching.

    Returns:
      cropped image,
      retained area fraction (1.0 means no crop).
    """
    width, height = image.size
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 1e-9:
        return image.copy(), 1.0

    original_area = width * height

    if current_ratio > target_ratio:
        new_width = int(round(height * target_ratio))
        left = max(0, (width - new_width) // 2)
        cropped = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(round(width / target_ratio))
        top = max(0, (height - new_height) // 2)
        cropped = image.crop((0, top, width, top + new_height))

    retained = (cropped.width * cropped.height) / original_area
    return cropped, retained


def resize_long_edge(
    image: Image.Image,
    long_edge: int,
    allow_upscale: bool = False,
) -> Image.Image:
    width, height = image.size
    current = max(width, height)

    if current == long_edge:
        return image.copy()

    if current < long_edge and not allow_upscale:
        return image.copy()

    scale = long_edge / current
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def save_inline_safe_png_or_jpeg(
    image: Image.Image,
    preferred_path: Path,
) -> Path:
    """
    Save as PNG first. If the resulting inline payload exceeds 7 MB,
    fall back to high-quality JPEG.
    """
    preferred_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = preferred_path.with_suffix(".png")
    image.save(png_path, format="PNG", optimize=True)

    if png_path.stat().st_size <= INLINE_IMAGE_MAX_BYTES:
        return png_path

    jpg_path = preferred_path.with_suffix(".jpg")
    image.save(
        jpg_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )

    if jpg_path.stat().st_size > INLINE_IMAGE_MAX_BYTES:
        raise RuntimeError(
            f"Prepared input vẫn >7 MB: {jpg_path} "
            f"({human_bytes(jpg_path.stat().st_size)})."
        )

    png_path.unlink(missing_ok=True)
    return jpg_path


def prepare_source_assets(
    source_path: Path,
    source_root: Path,
    allowed_ratios: frozenset[str],
    mode: str,
) -> dict[str, Any]:
    """
    Prepare normalized source assets for one benchmark image.

    practical:
      - Preserve the normalized source at its available resolution.
      - Send the SAME prepared input bytes to every model/target.
      - This matches the production question: given one ordinary image, which
        model gives the best faithful 1K/2K/4K result for the money?

    scientific:
      - Create shared synthetic low-resolution inputs (512/1024 long-edge).
      - Useful when the source is genuinely high resolution and can serve as
        native ground truth for the generated dimensions.
    """
    if mode not in {"practical", "scientific"}:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    source = open_rgb(source_path)
    source_width, source_height = source.size

    aspect_label, aspect_value = closest_supported_aspect_ratio(
        source_width,
        source_height,
        allowed_ratios,
    )

    normalized, retained_fraction = center_crop_to_ratio(
        source,
        aspect_value,
    )

    reference_dir = source_root / "reference"
    inputs_dir = source_root / "inputs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = reference_dir / "reference_normalized.png"
    normalized.save(normalized_path, format="PNG")

    input_paths: dict[Any, Path] = {}

    if mode == "practical":
        practical_input_path = save_inline_safe_png_or_jpeg(
            normalized,
            inputs_dir / "input_source",
        )
        input_paths["source"] = practical_input_path
        input_strategy = (
            "same normalized source image sent to every model/target"
        )
    else:
        for long_edge in sorted(set(INPUT_LONG_EDGE_BY_TARGET.values())):
            prepared = resize_long_edge(
                normalized,
                long_edge,
                allow_upscale=False,
            )
            input_paths[long_edge] = save_inline_safe_png_or_jpeg(
                prepared,
                inputs_dir / f"input_{long_edge}",
            )
        practical_input_path = None
        input_strategy = (
            "synthetic shared low-resolution inputs: 512px for 1K; "
            "1024px for 2K/4K"
        )

    return {
        "source": source,
        "normalized": normalized,
        "normalized_path": normalized_path,
        "input_paths": input_paths,
        "practical_input_path": practical_input_path,
        "benchmark_mode": mode,
        "input_strategy": input_strategy,
        "aspect_label": aspect_label,
        "aspect_value": aspect_value,
        "source_width": source_width,
        "source_height": source_height,
        "normalized_width": normalized.width,
        "normalized_height": normalized.height,
        "crop_retained_fraction": retained_fraction,
        "resolution_class": classify_resolution(source_width, source_height),
        "source_sha256": sha256_file(source_path),
    }

def make_ground_truth(
    normalized_source: Image.Image,
    output_width: int,
    output_height: int,
    gt_path: Path,
    mode: str,
) -> str:
    """
    Resize the normalized source to EXACT output dimensions.

    scientific mode:
      native_downscale = the original normalized source contains at least the
      generated dimensions on both axes and therefore provides high-confidence
      native ground truth.

    practical mode:
      native_reference = the ordinary source already contains at least the
      generated dimensions on both axes.
      upscaled_reference = the ordinary source must itself be enlarged to the
      generated dimensions. Metrics are useful as fidelity diagnostics only.
    """
    native = (
        normalized_source.width >= output_width
        and normalized_source.height >= output_height
    )

    gt = normalized_source.resize(
        (output_width, output_height),
        Image.Resampling.LANCZOS,
    )

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt.save(gt_path, format="PNG")

    if native:
        return "native_downscale" if mode == "scientific" else "native_reference"
    return "upscaled_reference"


# ============================================================
# RESPONSE / USAGE EXTRACTION
# ============================================================

def attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def token_count_for_modality(items: Any, modality_name: str) -> int:
    total = 0
    target = modality_name.upper()

    for item in items or []:
        modality = str(attr(item, "modality", "")).upper()
        if target not in modality:
            continue

        value = attr(item, "token_count", None)
        if value is None:
            value = attr(item, "tokens", 0)

        total += int(value or 0)

    return total


def normalize_traffic_type(value: Any) -> str:
    """Normalize SDK enum/string traffic type to e.g. ON_DEMAND_FLEX."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return {
            "prompt_tokens": 0,
            "input_text_tokens": 0,
            "input_image_tokens": 0,
            "candidate_tokens_total": 0,
            "output_text_tokens": 0,
            "output_image_tokens_reported": 0,
            "thought_tokens": 0,
            "cached_tokens": 0,
            "tool_use_prompt_tokens": 0,
            "total_tokens": 0,
            "traffic_type": "",
        }

    prompt_details = attr(usage, "prompt_tokens_details", [])
    candidate_details = attr(usage, "candidates_tokens_details", [])

    return {
        "prompt_tokens": int(attr(usage, "prompt_token_count", 0) or 0),
        "input_text_tokens": token_count_for_modality(prompt_details, "TEXT"),
        "input_image_tokens": token_count_for_modality(prompt_details, "IMAGE"),
        "candidate_tokens_total": int(
            attr(usage, "candidates_token_count", 0) or 0
        ),
        "output_text_tokens": token_count_for_modality(
            candidate_details,
            "TEXT",
        ),
        "output_image_tokens_reported": token_count_for_modality(
            candidate_details,
            "IMAGE",
        ),
        "thought_tokens": int(
            attr(usage, "thoughts_token_count", 0) or 0
        ),
        "cached_tokens": int(
            attr(usage, "cached_content_token_count", 0) or 0
        ),
        "tool_use_prompt_tokens": int(
            attr(usage, "tool_use_prompt_token_count", 0) or 0
        ),
        "total_tokens": int(attr(usage, "total_token_count", 0) or 0),
        "traffic_type": normalize_traffic_type(
            attr(usage, "traffic_type", None)
            or attr(usage, "trafficType", "")
        ),
    }


def decode_inline_data(data: Any) -> bytes:
    if data is None:
        raise ValueError("inline_data.data is empty.")

    if isinstance(data, bytes):
        return data

    if isinstance(data, str):
        return base64.b64decode(data)

    try:
        return bytes(data)
    except Exception as exc:
        raise TypeError(
            f"Không decode được inline image data: {type(data)}"
        ) from exc


def iter_response_parts(response: Any):
    seen: set[int] = set()

    # Canonical candidates path.
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if id(part) not in seen:
                seen.add(id(part))
                yield part

    # SDK convenience accessor fallback.
    try:
        for part in response.parts or []:
            if id(part) not in seen:
                seen.add(id(part))
                yield part
    except Exception:
        pass


def save_generated_image_from_response(
    response: Any,
    output_stem: Path,
) -> tuple[Path, str]:
    """
    Preserve the exact bytes returned by the model and use the actual MIME type
    to choose the extension.
    """
    for part in iter_response_parts(response):
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue

        data = getattr(inline, "data", None)
        if data is None:
            continue

        raw = decode_inline_data(data)
        mime = (getattr(inline, "mime_type", None) or OUTPUT_MIME_TYPE).lower()
        extension = extension_for_mime(mime)

        output_path = output_stem.with_suffix(extension)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)

        # Validate image bytes immediately.
        with Image.open(output_path) as img:
            img.verify()

        return output_path, mime

    raise RuntimeError("Model không trả về image output.")


# ============================================================
# COST
# ============================================================

def estimate_cost(
    model_cfg: ModelConfig,
    target: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    """
    Best-effort Flex PayGo cost estimate.

    Input:
      prompt_token_count * input token rate

    Output image:
      Prefer reported IMAGE modality tokens when available.
      Otherwise use the model's documented tokens for the requested size.

    Text/thinking:
      Only TEXT candidate tokens + thought tokens use the text/thinking rate.
      IMAGE candidate tokens are NOT charged again as text.
    """
    input_cost = (
        usage["prompt_tokens"]
        / 1_000_000
        * model_cfg.input_per_1m
    )

    output_image_tokens = (
        usage["output_image_tokens_reported"]
        if usage["output_image_tokens_reported"] > 0
        else model_cfg.expected_image_tokens[target]
    )

    output_image_tokens_source = (
        "usage_metadata"
        if usage["output_image_tokens_reported"] > 0
        else "pricing_fallback"
    )

    image_output_cost = (
        output_image_tokens
        / 1_000_000
        * model_cfg.image_output_per_1m
    )

    text_thinking_tokens = (
        usage["output_text_tokens"]
        + usage["thought_tokens"]
    )

    text_thinking_cost = (
        text_thinking_tokens
        / 1_000_000
        * model_cfg.text_thinking_output_per_1m
    )

    estimated_total = (
        input_cost
        + image_output_cost
        + text_thinking_cost
    )

    return {
        "input_cost_usd": input_cost,
        "text_thinking_cost_usd": text_thinking_cost,
        "image_output_cost_usd": image_output_cost,
        "image_output_tokens_used_for_cost": output_image_tokens,
        "image_output_tokens_cost_source": output_image_tokens_source,
        "estimated_total_cost_usd": estimated_total,
        "estimated_total_cost_vnd": (
            estimated_total * USD_TO_VND
            if USD_TO_VND > 0
            else math.nan
        ),
    }


def fixed_output_cost_for_target(
    cfg: ModelConfig,
    target: str,
) -> float:
    tokens = cfg.expected_image_tokens[target]
    return tokens / 1_000_000 * cfg.image_output_per_1m


# ============================================================
# METRICS
# ============================================================

def load_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img)


def laplacian_sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_metrics(
    output_path: Path,
    gt_path: Path,
) -> dict[str, float]:
    output = load_rgb_array(output_path)
    gt = load_rgb_array(gt_path)

    if output.shape != gt.shape:
        # Should not normally happen because GT is created from actual output dims.
        gt = cv2.resize(
            gt,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    psnr = peak_signal_noise_ratio(
        gt,
        output,
        data_range=255,
    )

    ssim = structural_similarity(
        gt,
        output,
        channel_axis=2,
        data_range=255,
    )

    output_sharpness = laplacian_sharpness(output)
    gt_sharpness = laplacian_sharpness(gt)

    mae = float(
        np.mean(
            np.abs(
                output.astype(np.float32)
                - gt.astype(np.float32)
            )
        )
    )

    sharpness_ratio_vs_gt = (
        output_sharpness / gt_sharpness
        if gt_sharpness > 0
        else math.nan
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "sharpness_laplacian": output_sharpness,
        "gt_sharpness_laplacian": gt_sharpness,
        "sharpness_ratio_vs_gt": sharpness_ratio_vs_gt,
        "rgb_mae": mae,
    }


# ============================================================
# API GENERATION
# ============================================================

def create_request_content(
    input_path: Path,
) -> types.Content:
    image_bytes = input_path.read_bytes()

    if len(image_bytes) > INLINE_IMAGE_MAX_BYTES:
        raise RuntimeError(
            f"Input inline >7 MB: {input_path} "
            f"({human_bytes(len(image_bytes))})"
        )

    return types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_for_path(input_path),
            ),
            types.Part.from_text(text=PROMPT),
        ],
    )


class GenerateRequestError(RuntimeError):
    """Raised after a model request fails or exhausts the retry policy."""

    def __init__(
        self,
        message: str,
        *,
        last_exception: Exception,
        attempts_used: int,
        retry_wait_sec: float,
        pre_request_delay_sec: float,
        api_attempts_total_sec: float,
        end_to_end_latency_sec: float,
    ) -> None:
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts_used = attempts_used
        self.retry_wait_sec = retry_wait_sec
        self.pre_request_delay_sec = pre_request_delay_sec
        self.api_attempts_total_sec = api_attempts_total_sec
        self.end_to_end_latency_sec = end_to_end_latency_sec


def get_http_status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    try:
        if code is not None:
            return int(code)
    except (TypeError, ValueError):
        pass

    text = str(exc).lower()
    for candidate in sorted(RETRYABLE_HTTP_CODES):
        if str(candidate) in text:
            return candidate
    return None


def is_retryable_exception(exc: Exception) -> bool:
    code = get_http_status_code(exc)
    if code in RETRYABLE_HTTP_CODES:
        return True

    text = str(exc).lower()
    retry_markers = (
        "resource_exhausted",
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
    )
    return any(marker in text for marker in retry_markers)


def retry_delay_seconds(exc: Exception, failed_attempt: int) -> float:
    """
    Return backoff after `failed_attempt` (1-based), with ±20% jitter.

    429 / RESOURCE_EXHAUSTED receives the slower schedule because the observed
    NB2 4K failures are consistent with transient Standard PayGo capacity/quota
    pressure. Other retryable server errors use a shorter schedule.
    """
    code = get_http_status_code(exc)
    text = str(exc).lower()

    if code == 429 or "resource_exhausted" in text:
        schedule = RETRY_429_SCHEDULE_SECONDS
    else:
        schedule = RETRY_5XX_SCHEDULE_SECONDS

    index = min(max(failed_attempt - 1, 0), len(schedule) - 1)
    base = schedule[index]
    jitter = random.uniform(
        1.0 - RETRY_JITTER_RATIO,
        1.0 + RETRY_JITTER_RATIO,
    )
    return max(0.0, base * jitter)


def pre_request_delay_seconds(
    model_cfg: ModelConfig,
    target_size: str,
    nb2_4k_pre_delay: float,
) -> float:
    if model_cfg.key == "nb2" and target_size == "4K":
        return max(0.0, nb2_4k_pre_delay)
    return 0.0


def generate(
    client: genai.Client,
    model_cfg: ModelConfig,
    input_path: Path,
    target_size: str,
    aspect_ratio: str,
    output_stem: Path,
    *,
    max_attempts: int,
    nb2_4k_pre_delay: float,
) -> tuple[Any, Path, str, float, dict[str, int], dict[str, float | int]]:
    content = create_request_content(input_path)

    config = types.GenerateContentConfig(
        response_modalities=[types.Modality.IMAGE],
        temperature=TEMPERATURE,
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=target_size,
            output_mime_type=OUTPUT_MIME_TYPE,
            output_compression_quality=OUTPUT_JPEG_QUALITY,
        ),
        # No tools are used; explicitly disable AFC to avoid irrelevant SDK warnings.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        labels={
            "workload": "image-upscale-benchmark",
            "model": model_cfg.key,
            "target": target_size.lower(),
        },
    )

    max_attempts = max(1, int(max_attempts))
    pre_delay = pre_request_delay_seconds(
        model_cfg,
        target_size,
        nb2_4k_pre_delay,
    )

    end_to_end_started = time.perf_counter()

    if pre_delay > 0:
        print(
            f"  Pre-request cooldown: {pre_delay:.1f}s "
            f"for {model_cfg.name} {target_size}"
        )
        time.sleep(pre_delay)

    retry_wait_total = 0.0
    api_attempts_total = 0.0
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()

        try:
            response = client.models.generate_content(
                model=model_cfg.model_id,
                contents=content,
                config=config,
            )
            final_attempt_latency = time.perf_counter() - attempt_started
            api_attempts_total += final_attempt_latency

            usage = extract_usage(response)

            # Safety against silent mis-routing / wrong billing assumptions.
            # Flex responses should report trafficType=ON_DEMAND_FLEX.
            traffic_type = str(usage.get("traffic_type", "")).upper()
            if traffic_type != FLEX_TRAFFIC_TYPE_REQUIRED:
                raise RuntimeError(
                    "Flex routing verification failed: expected "
                    f"{FLEX_TRAFFIC_TYPE_REQUIRED}, got {traffic_type or 'MISSING'}. "
                    "Refusing to label this request as Flex or estimate it with "
                    "Flex pricing. Check google-genai version and request headers."
                )

            # Save the generated bytes only after Flex routing is verified.
            output_path, output_mime = save_generated_image_from_response(
                response,
                output_stem,
            )

            end_to_end_latency = time.perf_counter() - end_to_end_started

            retry_stats: dict[str, float | int] = {
                "attempts_used": attempt,
                "retry_count": attempt - 1,
                "retry_wait_sec": retry_wait_total,
                "pre_request_delay_sec": pre_delay,
                "api_attempts_total_sec": api_attempts_total,
                "end_to_end_latency_sec": end_to_end_latency,
            }

            # Keep latency_sec backward-compatible: raw latency of the successful
            # final API attempt. End-to-end latency is reported separately.
            return (
                response,
                output_path,
                output_mime,
                final_attempt_latency,
                usage,
                retry_stats,
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            attempt_elapsed = time.perf_counter() - attempt_started
            api_attempts_total += attempt_elapsed
            last_exception = exc

            retryable = is_retryable_exception(exc)
            code = get_http_status_code(exc)

            if not retryable or attempt >= max_attempts:
                end_to_end_latency = time.perf_counter() - end_to_end_started
                raise GenerateRequestError(
                    (
                        f"Request failed after {attempt}/{max_attempts} attempt(s): "
                        f"{exc}"
                    ),
                    last_exception=exc,
                    attempts_used=attempt,
                    retry_wait_sec=retry_wait_total,
                    pre_request_delay_sec=pre_delay,
                    api_attempts_total_sec=api_attempts_total,
                    end_to_end_latency_sec=end_to_end_latency,
                ) from exc

            wait_seconds = retry_delay_seconds(exc, attempt)
            retry_wait_total += wait_seconds

            print(
                f"  Retryable error on attempt {attempt}/{max_attempts} "
                f"(HTTP {code or 'unknown'}): {exc}"
            )
            print(
                f"  Waiting {wait_seconds:.1f}s before attempt "
                f"{attempt + 1}/{max_attempts}..."
            )
            time.sleep(wait_seconds)

    # Defensive guard; the loop must either return or raise.
    assert last_exception is not None
    raise last_exception


# ============================================================
# PLAN / FILTERING
# ============================================================

def build_test_matrix(
    selected_models: list[str],
    selected_targets: list[str],
) -> list[tuple[str, ModelConfig, str]]:
    matrix: list[tuple[str, ModelConfig, str]] = []
    target_set = set(selected_targets)

    for model_key in selected_models:
        cfg = MODELS[model_key]
        for target in cfg.sizes:
            if target in target_set:
                matrix.append((model_key, cfg, target))

    if not matrix:
        raise RuntimeError("Matrix rỗng sau khi lọc model/target.")

    return matrix


def output_only_cost_per_source_run(
    matrix: list[tuple[str, ModelConfig, str]],
) -> float:
    return sum(
        fixed_output_cost_for_target(cfg, target)
        for _, cfg, target in matrix
    )


def print_run_plan(
    sources: list[Path],
    matrix: list[tuple[str, ModelConfig, str]],
    runs: int,
    *,
    mode: str,
    max_attempts: int,
    request_delay: float,
    nb2_4k_pre_delay: float,
) -> None:
    total_requests = len(sources) * len(matrix) * runs
    output_only = (
        output_only_cost_per_source_run(matrix)
        * len(sources)
        * runs
    )

    print("=" * 78)
    print("GEMINI IMAGE UPSCALE BENCHMARK — FLEX PAYGO / AGENT PLATFORM")
    print("=" * 78)
    print(f"Project:        {PROJECT_ID}")
    print(f"Location:       {LOCATION}")
    print(f"Dataset:        {DATASET_DIR}")
    print(f"Source images:  {len(sources)}")
    print(f"Runs/test:      {runs}")
    print(f"Requests/run:   {len(matrix)}")
    print(f"Total requests: {total_requests}")
    print(f"Pricing date:   {PRICING_SNAPSHOT_DATE}")
    print("Consumption:    Flex PayGo (required traffic=ON_DEMAND_FLEX)")
    print(f"Mode:           {mode}")
    if mode == "practical":
        print("Input strategy: same normalized source image for every target")
    else:
        print("Input strategy: 512px -> 1K; 1024px -> 2K/4K")
    print(f"Max attempts:   {max_attempts}")
    print(f"Case delay:     {max(0.0, request_delay):.1f}s")
    print(f"NB2 4K prewait: {max(0.0, nb2_4k_pre_delay):.1f}s")
    print()
    print("Output-only cost preview:")
    print(f"  USD: {money_usd(output_only)}")
    if USD_TO_VND > 0:
        print(f"  VND: {money_vnd(output_only)}")
    print("  + input tokens + possible thinking/text output.")
    print()

    print("Matrix:")
    for _, cfg, target in matrix:
        fixed = fixed_output_cost_for_target(cfg, target)
        suffix = f" (~{money_vnd(fixed)})" if USD_TO_VND > 0 else ""
        preview = (
            "Preview output feature"
            if target == "4K" and cfg.key in {"nb2", "pro"}
            else ""
        )
        print(
            f"  - {cfg.name:<24} {target:<2} "
            f"output≈{money_usd(fixed)} {preview}{suffix}"
        )

    print()
    print("Sources:")
    for source in sources:
        try:
            img = open_rgb(source)
            print(
                f"  - {source.name}: "
                f"{img.width}x{img.height} "
                f"({classify_resolution(img.width, img.height)})"
            )
        except Exception as exc:
            print(f"  - {source.name}: ERROR: {exc}")

    print("=" * 78)


# ============================================================
# REPORT HELPERS
# ============================================================

def relative_uri(path_value: Any, run_root: Path) -> str:
    if not path_value:
        return ""
    try:
        return os.path.relpath(
            str(path_value),
            str(run_root),
        ).replace("\\", "/")
    except Exception:
        return ""


def image_cell(
    path_value: Any,
    run_root: Path,
    width: int = 180,
) -> str:
    if not path_value:
        return ""
    uri = relative_uri(path_value, run_root)
    escaped = html.escape(uri)
    return (
        f'<a href="{escaped}" target="_blank">'
        f'<img src="{escaped}" width="{width}"></a>'
    )


def html_money_vnd(value: Any) -> str:
    if USD_TO_VND <= 0:
        return ""
    number = safe_float(value)
    if math.isnan(number):
        return ""
    return f"{number:,.0f} VND"


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful benchmark rows without mixing away GT confidence."""
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(
            ["model_key", "model_name", "model_id", "target"],
            as_index=False,
        )
        .agg(
            requests=("status", "count"),
            native_rows=(
                "gt_quality",
                lambda s: int(
                    sum(
                        value in {"native_downscale", "native_reference"}
                        for value in s
                    )
                ),
            ),
            upscaled_reference_rows=(
                "gt_quality",
                lambda s: int(sum(value == "upscaled_reference" for value in s)),
            ),
            avg_latency_sec=("latency_sec", "mean"),
            avg_end_to_end_latency_sec=("end_to_end_latency_sec", "mean"),
            avg_attempts_used=("attempts_used", "mean"),
            total_retries=("retry_count", "sum"),
            avg_retry_wait_sec=("retry_wait_sec", "mean"),
            avg_estimated_total_cost_usd=(
                "estimated_total_cost_usd",
                "mean",
            ),
            total_estimated_cost_usd=(
                "estimated_total_cost_usd",
                "sum",
            ),
            avg_estimated_total_cost_vnd=(
                "estimated_total_cost_vnd",
                "mean",
            ),
            total_estimated_cost_vnd=(
                "estimated_total_cost_vnd",
                "sum",
            ),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_input_image_tokens=("input_image_tokens", "mean"),
            avg_thought_tokens=("thought_tokens", "mean"),
            avg_output_image_tokens=(
                "output_image_tokens_reported",
                "mean",
            ),
            avg_ssim=("ssim", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_sharpness_laplacian=(
                "sharpness_laplacian",
                "mean",
            ),
            avg_sharpness_ratio_vs_gt=(
                "sharpness_ratio_vs_gt",
                "mean",
            ),
            avg_rgb_mae=("rgb_mae", "mean"),
        )
        .sort_values(["target", "avg_estimated_total_cost_usd"])
        .reset_index(drop=True)
    )

    summary["metric_confidence"] = np.where(
        summary["upscaled_reference_rows"] == 0,
        "high_native_reference",
        np.where(
            summary["native_rows"] == 0,
            "diagnostic_upscaled_reference",
            "mixed_reference_quality",
        ),
    )

    return summary


def build_recommendations(
    ok: pd.DataFrame,
    benchmark_mode: str,
) -> pd.DataFrame:
    """
    Build a conservative automatic value recommendation per target.

    This is intentionally NOT presented as a final visual-quality verdict.
    Rule:
      1. Average each model's successful rows for the target.
      2. Find the best average SSIM.
      3. Treat models within 0.01 absolute SSIM of the best as "near-best".
      4. Among near-best models, choose the lowest average cost.

    If SSIM is unavailable, choose the lowest-cost successful model.

    When any row uses upscaled_reference, recommendation confidence is
    "provisional" because pixel metrics compare against an enlarged source
    reference rather than native high-resolution truth.
    """
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    target_order = {"1K": 1, "2K": 2, "4K": 3}

    for target in sorted(
        ok["target"].dropna().unique(),
        key=lambda value: target_order.get(str(value), 99),
    ):
        target_rows = ok[ok["target"] == target].copy()

        grouped = (
            target_rows.groupby(
                ["model_key", "model_name", "model_id"],
                as_index=False,
            )
            .agg(
                requests=("status", "count"),
                avg_cost_usd=("estimated_total_cost_usd", "mean"),
                avg_e2e_latency_sec=("end_to_end_latency_sec", "mean"),
                avg_ssim=("ssim", "mean"),
                avg_psnr=("psnr", "mean"),
                avg_rgb_mae=("rgb_mae", "mean"),
                total_retries=("retry_count", "sum"),
                native_rows=(
                    "gt_quality",
                    lambda s: int(
                        sum(
                            value in {"native_downscale", "native_reference"}
                            for value in s
                        )
                    ),
                ),
                upscaled_reference_rows=(
                    "gt_quality",
                    lambda s: int(
                        sum(value == "upscaled_reference" for value in s)
                    ),
                ),
            )
        )

        if grouped.empty:
            continue

        cheapest = grouped.loc[grouped["avg_cost_usd"].idxmin()]
        fastest = grouped.loc[grouped["avg_e2e_latency_sec"].idxmin()]

        valid_ssim = grouped[grouped["avg_ssim"].notna()].copy()
        if valid_ssim.empty:
            chosen = cheapest
            best_fidelity = None
            rule = "No SSIM available; selected lowest-cost successful model."
        else:
            best_idx = valid_ssim["avg_ssim"].idxmax()
            best_fidelity = valid_ssim.loc[best_idx]
            best_ssim = float(best_fidelity["avg_ssim"])
            near_best = valid_ssim[
                valid_ssim["avg_ssim"] >= best_ssim - 0.01
            ].copy()
            chosen = near_best.loc[near_best["avg_cost_usd"].idxmin()]
            rule = (
                "Lowest cost among models within 0.01 SSIM of the best "
                "average SSIM."
            )

        has_upscaled = int(target_rows["gt_quality"].eq("upscaled_reference").sum())
        has_native = int(
            target_rows["gt_quality"].isin(
                {"native_downscale", "native_reference"}
            ).sum()
        )

        if has_upscaled == 0:
            confidence = "high"
            basis = "native reference metrics"
        elif has_native == 0:
            confidence = "provisional"
            basis = "upscaled-reference diagnostics + cost/latency"
        else:
            confidence = "mixed"
            basis = "mixed native/upscaled references"

        rows.append({
            "target": target,
            "benchmark_mode": benchmark_mode,
            "recommended_model_key": chosen["model_key"],
            "recommended_model": chosen["model_name"],
            "recommended_price_per_image_usd": float(chosen["avg_cost_usd"]),
            "recommended_avg_e2e_latency_sec": float(
                chosen["avg_e2e_latency_sec"]
            ),
            "recommended_avg_ssim": safe_float(chosen["avg_ssim"]),
            "recommended_avg_psnr": safe_float(chosen["avg_psnr"]),
            "recommended_avg_rgb_mae": safe_float(chosen["avg_rgb_mae"]),
            "recommended_total_retries": int(chosen["total_retries"]),
            "cheapest_model": cheapest["model_name"],
            "cheapest_price_per_image_usd": float(cheapest["avg_cost_usd"]),
            "fastest_model": fastest["model_name"],
            "fastest_avg_e2e_latency_sec": float(
                fastest["avg_e2e_latency_sec"]
            ),
            "best_fidelity_model_by_ssim": (
                best_fidelity["model_name"]
                if best_fidelity is not None
                else ""
            ),
            "best_avg_ssim": (
                safe_float(best_fidelity["avg_ssim"])
                if best_fidelity is not None
                else math.nan
            ),
            "confidence": confidence,
            "metric_basis": basis,
            "selection_rule": rule,
        })

    return pd.DataFrame(rows)


def summary_table_html(
    summary: pd.DataFrame,
    title: str,
    note: str = "",
) -> str:
    if summary.empty:
        return ""

    rows: list[str] = []

    for _, row in summary.iterrows():
        cost_vnd = (
            html_money_vnd(
                safe_float(row["avg_estimated_total_cost_usd"]) * USD_TO_VND
            )
            if USD_TO_VND > 0
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_name']))}</td>"
            f"<td>{html.escape(str(row['target']))}</td>"
            f"<td>{int(row['requests'])}</td>"
            f"<td>{int(row['native_rows'])}</td>"
            f"<td>{int(row['upscaled_reference_rows'])}</td>"
            f"<td>{safe_num(row['avg_latency_sec'], 2)}s</td>"
            f"<td>{safe_num(row['avg_end_to_end_latency_sec'], 2)}s</td>"
            f"<td>{safe_num(row['avg_attempts_used'], 2)}</td>"
            f"<td>{int(row['total_retries'])}</td>"
            f"<td>${safe_num(row['avg_estimated_total_cost_usd'], 6)}</td>"
            f"<td>{cost_vnd}</td>"
            f"<td>{safe_num(row['avg_ssim'], 4)}</td>"
            f"<td>{safe_num(row['avg_psnr'], 2)}</td>"
            f"<td>{safe_num(row['avg_sharpness_laplacian'], 2)}</td>"
            f"<td>{safe_num(row['avg_sharpness_ratio_vs_gt'], 3)}</td>"
            f"<td>{safe_num(row['avg_rgb_mae'], 2)}</td>"
            f"<td>{html.escape(str(row['metric_confidence']))}</td>"
            "</tr>"
        )

    note_html = (
        f"<p><small>{html.escape(note)}</small></p>"
        if note
        else ""
    )

    return f"""
    <h2>{html.escape(title)}</h2>
    {note_html}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Target</th>
            <th>Requests</th>
            <th>Native rows</th>
            <th>Upscaled-ref rows</th>
            <th>Avg final-attempt latency</th>
            <th>Avg end-to-end latency</th>
            <th>Avg attempts</th>
            <th>Total retries</th>
            <th>Avg estimated cost USD</th>
            <th>Avg estimated cost VND</th>
            <th>Avg SSIM ↑</th>
            <th>Avg PSNR ↑</th>
            <th>Avg sharpness</th>
            <th>Sharpness / GT</th>
            <th>Avg RGB MAE ↓</th>
            <th>Metric confidence</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def recommendations_table_html(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return ""

    rows: list[str] = []

    for _, row in recommendations.iterrows():
        price = safe_float(row["recommended_price_per_image_usd"])
        price_vnd = money_vnd(price) if USD_TO_VND > 0 else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['target']))}</td>"
            f"<td><strong>{html.escape(str(row['recommended_model']))}</strong></td>"
            f"<td>${safe_num(price, 6)}</td>"
            f"<td>{html.escape(price_vnd)}</td>"
            f"<td>{safe_num(row['recommended_avg_e2e_latency_sec'], 2)}s</td>"
            f"<td>{safe_num(row['recommended_avg_ssim'], 4)}</td>"
            f"<td>{safe_num(row['recommended_avg_psnr'], 2)}</td>"
            f"<td>{safe_num(row['recommended_avg_rgb_mae'], 2)}</td>"
            f"<td>{html.escape(str(row['confidence']))}</td>"
            f"<td>{html.escape(str(row['metric_basis']))}</td>"
            f"<td>{html.escape(str(row['selection_rule']))}</td>"
            "</tr>"
        )

    return f"""
    <h2>Automatic value recommendation</h2>
    <div class="warning">
      This table is an automatic cost/quality screening result, not a final
      visual-quality verdict. For practical low-resolution inputs, rows based on
      <code>upscaled_reference</code> are provisional and must be confirmed by
      manual fidelity/hallucination review.
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Target</th>
            <th>Auto value pick</th>
            <th>Price / image USD</th>
            <th>Approx VND</th>
            <th>Avg E2E latency</th>
            <th>Avg SSIM</th>
            <th>Avg PSNR</th>
            <th>Avg RGB MAE</th>
            <th>Confidence</th>
            <th>Metric basis</th>
            <th>Selection rule</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def create_html_report(
    df: pd.DataFrame,
    report_path: Path,
    run_root: Path,
    manifest: dict[str, Any],
) -> None:
    ok = df[df["status"] == "ok"].copy()

    total_estimated = (
        float(ok["estimated_total_cost_usd"].sum())
        if not ok.empty
        else 0.0
    )

    benchmark_mode = str(manifest.get("benchmark_mode", "scientific"))

    full_summary = aggregate_summary(df)

    native_ok = ok[
        ok["gt_quality"].isin({"native_downscale", "native_reference"})
    ].copy()
    native_summary = aggregate_summary(native_ok) if not native_ok.empty else pd.DataFrame()

    upscaled_ok = ok[
        ok["gt_quality"] == "upscaled_reference"
    ].copy()
    upscaled_summary = (
        aggregate_summary(upscaled_ok)
        if not upscaled_ok.empty
        else pd.DataFrame()
    )

    recommendations = build_recommendations(ok, benchmark_mode)

    overall_summary_html = summary_table_html(
        full_summary,
        "Aggregate summary — all successful rows",
        (
            "Use this table for cost, latency, retries, and broad screening. "
            "Do not interpret mixed/upscaled-reference quality metrics as native "
            "ground-truth accuracy."
        ),
    )

    native_summary_html = summary_table_html(
        native_summary,
        "High-confidence quality summary — native reference only",
        (
            "These rows have a source/reference that is at least as large as the "
            "generated output dimensions. SSIM/PSNR/MAE are the strongest objective "
            "quality evidence available in this benchmark."
        ),
    )

    upscaled_summary_html = summary_table_html(
        upscaled_summary,
        "Practical diagnostic summary — upscaled reference only",
        (
            "These rows use an enlarged source reference. They are useful for "
            "real-world fidelity diagnostics, cost, latency and comparative screening, "
            "but they are not true high-resolution ground truth."
        ),
    )

    recommendation_html = recommendations_table_html(recommendations)

    detail_rows: list[str] = []

    for _, row in df.iterrows():
        if row["status"] == "ok":
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('benchmark_mode', benchmark_mode)))}</td>"
                f"<td>{html.escape(str(row.get('traffic_type', '')))}</td>"
                f"<td>{html.escape(str(row['source']))}</td>"
                f"<td>{html.escape(str(row['aspect_ratio']))}</td>"
                f"<td>{html.escape(str(row['model_name']))}</td>"
                f"<td>{html.escape(str(row['target']))}</td>"
                f"<td>{int(row['run'])}</td>"
                f"<td>{image_cell(row['input_path'], run_root)}</td>"
                f"<td>{image_cell(row['output_path'], run_root)}</td>"
                f"<td>{image_cell(row['gt_path'], run_root)}</td>"
                f"<td>{html.escape(str(row['input_dimensions']))}</td>"
                f"<td>{html.escape(str(row['output_dimensions']))}</td>"
                f"<td>{html.escape(str(row['gt_quality']))}</td>"
                f"<td>{safe_num(row['latency_sec'], 2)}s</td>"
                f"<td>{safe_num(row['end_to_end_latency_sec'], 2)}s</td>"
                f"<td>{safe_num(row['attempts_used'], 0)}</td>"
                f"<td>{safe_num(row['retry_wait_sec'], 2)}s</td>"
                f"<td>{safe_num(row['pre_request_delay_sec'], 2)}s</td>"
                f"<td>${safe_num(row['estimated_total_cost_usd'], 6)}</td>"
                f"<td>{html_money_vnd(row['estimated_total_cost_vnd'])}</td>"
                f"<td>{safe_num(row['prompt_tokens'], 0)}</td>"
                f"<td>{safe_num(row['input_image_tokens'], 0)}</td>"
                f"<td>{safe_num(row['thought_tokens'], 0)}</td>"
                f"<td>{safe_num(row['output_image_tokens_reported'], 0)}</td>"
                f"<td>{safe_num(row['ssim'], 4)}</td>"
                f"<td>{safe_num(row['psnr'], 2)}</td>"
                f"<td>{safe_num(row['sharpness_laplacian'], 2)}</td>"
                f"<td>{safe_num(row['sharpness_ratio_vs_gt'], 3)}</td>"
                f"<td>{safe_num(row['rgb_mae'], 2)}</td>"
                "</tr>"
            )
        else:
            detail_rows.append(
                "<tr class='error'>"
                f"<td>{html.escape(str(row.get('benchmark_mode', benchmark_mode)))}</td>"
                f"<td>{html.escape(str(row.get('traffic_type', '')))}</td>"
                f"<td>{html.escape(str(row['source']))}</td>"
                f"<td>{html.escape(str(row['aspect_ratio']))}</td>"
                f"<td>{html.escape(str(row['model_name']))}</td>"
                f"<td>{html.escape(str(row['target']))}</td>"
                f"<td>{int(row['run'])}</td>"
                f"<td colspan='22'>{html.escape(str(row['error']))}</td>"
                "</tr>"
            )

    total_vnd = (
        f"{total_estimated * USD_TO_VND:,.0f} VND"
        if USD_TO_VND > 0
        else "N/A (set USD_TO_VND in .env)"
    )

    if benchmark_mode == "practical":
        matrix_text = (
            "Same normalized source image → each requested 1K / 2K / 4K target."
        )
        mode_explanation = (
            "Practical mode answers the production question: given one ordinary "
            "input image, which model produces the most useful faithful upscale "
            "for its price and latency?"
        )
    else:
        matrix_text = "512px → 1K; 1024px → 2K; 1024px → 4K."
        mode_explanation = (
            "Scientific mode synthesizes low-resolution inputs from a higher-resolution "
            "source so native high-resolution ground truth can be measured."
        )

    report_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Gemini Image Upscale Benchmark — Flex PayGo</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 24px;
          color: #222;
          line-height: 1.45;
        }}
        .note {{
          background: #f5f7fb;
          padding: 12px 16px;
          border-radius: 8px;
          margin: 12px 0;
        }}
        .warning {{
          background: #fff4e5;
          padding: 12px 16px;
          border-radius: 8px;
          margin: 12px 0;
        }}
        .cost {{
          background: #eef9f0;
          padding: 12px 16px;
          border-radius: 8px;
          margin: 12px 0;
        }}
        .table-wrap {{
          overflow-x: auto;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin: 16px 0 32px;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 8px;
          vertical-align: top;
          text-align: left;
          white-space: nowrap;
        }}
        th {{
          background: #f0f2f5;
          position: sticky;
          top: 0;
          z-index: 1;
        }}
        img {{
          max-width: 180px;
          height: auto;
          display: block;
        }}
        .error {{
          background: #ffecec;
        }}
        code {{
          background: #f4f4f4;
          padding: 2px 5px;
          border-radius: 4px;
        }}
      </style>
    </head>
    <body>
      <h1>Gemini Image Upscale Benchmark — Flex PayGo</h1>

      <div class="note">
        <strong>Backend:</strong> Gemini Enterprise Agent Platform / Google Cloud<br>
        <strong>Consumption mode:</strong> Flex PayGo<br>
        <strong>Required traffic type:</strong> ON_DEMAND_FLEX<br>
        <strong>Project:</strong> {html.escape(PROJECT_ID)}<br>
        <strong>Location:</strong> {html.escape(LOCATION)}<br>
        <strong>Benchmark mode:</strong> {html.escape(benchmark_mode)}<br>
        <strong>Input strategy:</strong>
        {html.escape(str(manifest.get('input_strategy', '')))}<br>
        <strong>Mode purpose:</strong> {html.escape(mode_explanation)}<br>
        <strong>Pricing snapshot:</strong> {PRICING_SNAPSHOT_DATE}<br>
        <strong>Matrix:</strong> {html.escape(matrix_text)}<br>
        <strong>Retry policy:</strong>
        up to {manifest['retry_policy']['max_attempts']} attempts;
        429 backoff {html.escape(str(manifest['retry_policy']['retry_429_schedule_seconds']))};
        jitter ±{manifest['retry_policy']['retry_jitter_ratio'] * 100:.0f}%.<br>
        <strong>NB2 4K pre-request cooldown:</strong>
        {manifest['retry_policy']['nb2_4k_pre_request_delay_seconds']:.1f}s.
      </div>

      <div class="cost">
        <strong>Estimated successful-request cost:</strong>
        ${total_estimated:.6f} USD<br>
        <strong>Approx VND:</strong> {total_vnd}<br>
        <small>
          This is an estimate from usage metadata + pricing snapshot.
          Google Cloud Billing is authoritative for actual billed spend,
          credits, discounts, taxes, and rounding.
        </small>
      </div>

      <div class="warning">
        Flex PayGo is synchronous but latency-tolerant and can have longer response
        times and higher throttling than Standard PayGo. Every successful row in this
        report must show <code>ON_DEMAND_FLEX</code>; otherwise the script refuses to
        apply Flex pricing.<br><br>
        Gemini image models are generative editors, not deterministic classical
        upscalers. The requirement "same image, only sharper" is therefore evaluated
        as fidelity, not as a guarantee of bit-for-bit identity. SSIM/PSNR/MAE are
        diagnostic. Rows marked <code>upscaled_reference</code> do not have native
        high-resolution ground truth and require manual review for hallucination,
        identity, text/logo and color preservation.
      </div>

      {recommendation_html}
      {overall_summary_html}
      {native_summary_html}
      {upscaled_summary_html}

      <h2>Detailed results</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mode</th>
              <th>Traffic type</th>
              <th>Source</th>
              <th>Aspect</th>
              <th>Model</th>
              <th>Target</th>
              <th>Run</th>
              <th>Input</th>
              <th>Output</th>
              <th>Reference</th>
              <th>Input size</th>
              <th>Output size</th>
              <th>Reference quality</th>
              <th>Final attempt latency</th>
              <th>End-to-end latency</th>
              <th>Attempts</th>
              <th>Retry wait</th>
              <th>Pre-request wait</th>
              <th>Estimated cost USD</th>
              <th>Estimated cost VND</th>
              <th>Prompt tokens</th>
              <th>Input image tokens</th>
              <th>Thinking tokens</th>
              <th>Output image tokens</th>
              <th>SSIM ↑</th>
              <th>PSNR ↑</th>
              <th>Sharpness</th>
              <th>Sharpness / reference</th>
              <th>RGB MAE ↓</th>
            </tr>
          </thead>
          <tbody>{''.join(detail_rows)}</tbody>
        </table>
      </div>
    </body>
    </html>
    """

    report_path.write_text(report_html, encoding="utf-8")


def create_summary_csv(
    df: pd.DataFrame,
    path: Path,
    benchmark_mode: str,
) -> None:
    """
    Write:
      summary.csv             -> all successful rows
      summary_native.csv      -> native-reference rows only
      summary_reference.csv   -> upscaled-reference rows only
      recommendations.csv     -> automatic per-target value screening
    """
    ok = df[df["status"] == "ok"].copy()

    summary = aggregate_summary(df)
    summary.to_csv(path, index=False, encoding="utf-8-sig")

    native = ok[
        ok["gt_quality"].isin({"native_downscale", "native_reference"})
    ].copy()
    native_summary = (
        aggregate_summary(native)
        if not native.empty
        else pd.DataFrame()
    )
    native_summary.to_csv(
        path.parent / "summary_native.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference = ok[
        ok["gt_quality"] == "upscaled_reference"
    ].copy()
    reference_summary = (
        aggregate_summary(reference)
        if not reference.empty
        else pd.DataFrame()
    )
    reference_summary.to_csv(
        path.parent / "summary_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recommendations = build_recommendations(ok, benchmark_mode)
    recommendations.to_csv(
        path.parent / "recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MANIFEST
# ============================================================

def create_manifest(
    run_root: Path,
    sources: list[Path],
    matrix: list[tuple[str, ModelConfig, str]],
    args: argparse.Namespace,
    allowed_ratios: frozenset[str],
) -> dict[str, Any]:
    manifest = {
        "created_at_utc": utc_now_iso(),
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "backend": "Gemini Enterprise Agent Platform",
        "consumption_mode": "Flex PayGo",
        "required_traffic_type": FLEX_TRAFFIC_TYPE_REQUIRED,
        "flex_headers": FLEX_HEADERS,
        "api_version": "v1",
        "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
        "pricing_mode": "Flex/Batch discounted rates — Flex routing verified per response",
        "usd_to_vnd": USD_TO_VND if USD_TO_VND > 0 else None,
        "runs_per_test": args.runs,
        "benchmark_mode": args.mode,
        "input_strategy": (
            "same normalized source image sent to every model/target"
            if args.mode == "practical"
            else "synthetic 512px input for 1K and 1024px input for 2K/4K"
        ),
        "skip_metrics": args.skip_metrics,
        "temperature": TEMPERATURE,
        "output_mime_type": OUTPUT_MIME_TYPE,
        "output_jpeg_quality": OUTPUT_JPEG_QUALITY,
        "retry_policy": {
            "max_attempts": args.max_attempts,
            "retryable_http_codes": sorted(RETRYABLE_HTTP_CODES),
            "retry_429_schedule_seconds": list(RETRY_429_SCHEDULE_SECONDS),
            "retry_5xx_schedule_seconds": list(RETRY_5XX_SCHEDULE_SECONDS),
            "retry_jitter_ratio": RETRY_JITTER_RATIO,
            "request_delay_seconds": max(0.0, args.request_delay),
            "nb2_4k_pre_request_delay_seconds": max(0.0, args.nb2_4k_pre_delay),
        },
        "source_filters": args.sources,
        "prompt_sha256": hashlib.sha256(
            PROMPT.encode("utf-8")
        ).hexdigest(),
        "input_long_edge_by_target": INPUT_LONG_EDGE_BY_TARGET,
        "allowed_common_aspect_ratios": sorted(allowed_ratios),
        "sources": [
            {
                "name": p.name,
                "path": str(p),
                "sha256": sha256_file(p),
            }
            for p in sources
        ],
        "matrix": [
            {
                "model_key": key,
                "model_name": cfg.name,
                "model_id": cfg.model_id,
                "target": target,
                "input_per_1m": cfg.input_per_1m,
                "text_thinking_output_per_1m": (
                    cfg.text_thinking_output_per_1m
                ),
                "image_output_per_1m": cfg.image_output_per_1m,
                "expected_output_image_tokens": (
                    cfg.expected_image_tokens[target]
                ),
            }
            for key, cfg, target in matrix
        ],
    }

    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_root / "prompt.txt").write_text(
        PROMPT + "\n",
        encoding="utf-8",
    )

    return manifest


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_error(exc: Exception) -> str:
    root = (
        exc.last_exception
        if isinstance(exc, GenerateRequestError)
        else exc
    )
    text = str(root).lower()
    code = get_http_status_code(root)

    if code == 429 or "resource_exhausted" in text:
        return "resource_exhausted_429"
    if code == 403 or "permission_denied" in text:
        return "permission"
    if code == 400 or "invalid_argument" in text:
        return "invalid_request"
    if code in {408, 500, 502, 503, 504}:
        return f"retryable_http_{code}"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if isinstance(root, errors.APIError) and code:
        return f"api_{code}"
    return "other"


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs phải >= 1.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts phải >= 1.")
    if args.request_delay < 0:
        raise ValueError("--request-delay phải >= 0.")
    if args.nb2_4k_pre_delay < 0:
        raise ValueError("--nb2-4k-pre-delay phải >= 0.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    all_sources = discover_sources()
    sources = filter_sources(all_sources, args.sources)

    if not all_sources:
        raise RuntimeError(
            f"Không tìm thấy ảnh trong: {DATASET_DIR}\n"
            "Hãy bỏ ảnh vào folder dataset/."
        )

    if not sources:
        raise RuntimeError(
            "Không có source nào khớp --sources. "
            f"Available: {[p.name for p in all_sources]}"
        )

    matrix = build_test_matrix(
        args.models,
        args.targets,
    )

    allowed_ratios = ratio_intersection(args.models)

    print_run_plan(
        sources,
        matrix,
        args.runs,
        mode=args.mode,
        max_attempts=args.max_attempts,
        request_delay=args.request_delay,
        nb2_4k_pre_delay=args.nb2_4k_pre_delay,
    )

    if args.dry_run:
        print("DRY RUN: chưa gọi API.")
        return 0

    if not args.yes:
        answer = input(
            "Bắt đầu gọi API có tính phí? [y/N]: "
        ).strip().lower()

        if answer not in {"y", "yes"}:
            print("Đã hủy. Chưa gọi API.")
            return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_DIR / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest(
        run_root,
        sources,
        matrix,
        args,
        allowed_ratios,
    )

    client = build_client()
    results: list[dict[str, Any]] = []

    try:
        for source_path in sources:
            print()
            print("=" * 78)
            print("SOURCE:", source_path.name)
            print("=" * 78)

            source_root = run_root / source_path.stem
            source_root.mkdir(parents=True, exist_ok=True)

            try:
                assets = prepare_source_assets(
                    source_path,
                    source_root,
                    allowed_ratios,
                    args.mode,
                )
            except Exception as exc:
                print("ERROR PREPARING SOURCE:", exc)
                continue

            normalized = assets["normalized"]
            retained_pct = assets["crop_retained_fraction"] * 100

            print(
                f"Original:   "
                f"{assets['source_width']}x{assets['source_height']} "
                f"({assets['resolution_class']})"
            )
            print(
                f"Normalized: "
                f"{assets['normalized_width']}x{assets['normalized_height']} "
                f"| aspect={assets['aspect_label']} "
                f"| retained={retained_pct:.2f}%"
            )

            if retained_pct < 95:
                print(
                    "WARNING: nearest supported aspect ratio required "
                    f"{100 - retained_pct:.2f}% area crop."
                )

            for model_key, model_cfg, target in matrix:
                if args.mode == "practical":
                    input_path = assets["practical_input_path"]
                    if input_path is None:
                        raise RuntimeError("Missing practical input path.")
                else:
                    input_long_edge = INPUT_LONG_EDGE_BY_TARGET[target]
                    input_path = assets["input_paths"][input_long_edge]

                input_img = open_rgb(input_path)
                input_dimensions = (
                    f"{input_img.width}x{input_img.height}"
                )

                for run_number in range(1, args.runs + 1):
                    output_stem = (
                        source_root
                        / "outputs"
                        / model_key
                        / target
                        / f"run_{run_number:02d}"
                    )

                    print(
                        f"{model_cfg.name} | {target} | "
                        f"run {run_number}/{args.runs} | "
                        f"input {input_dimensions} | "
                        f"aspect {assets['aspect_label']}"
                    )

                    row: dict[str, Any] = {
                        "source": source_path.name,
                        "benchmark_mode": args.mode,
                        "input_strategy": assets["input_strategy"],
                        "source_sha256": assets["source_sha256"],
                        "source_width": assets["source_width"],
                        "source_height": assets["source_height"],
                        "source_resolution_class": (
                            assets["resolution_class"]
                        ),
                        "normalized_width": assets["normalized_width"],
                        "normalized_height": assets["normalized_height"],
                        "crop_retained_fraction": (
                            assets["crop_retained_fraction"]
                        ),
                        "aspect_ratio": assets["aspect_label"],
                        "model_key": model_key,
                        "model_name": model_cfg.name,
                        "model_id": model_cfg.model_id,
                        "target": target,
                        "run": run_number,
                        "input_path": str(input_path),
                        "input_dimensions": input_dimensions,
                        "input_file_bytes": input_path.stat().st_size,
                        "input_sha256": sha256_file(input_path),
                        "output_path": "",
                        "output_mime_type": "",
                        "output_dimensions": "",
                        "output_file_bytes": 0,
                        "output_sha256": "",
                        "gt_path": "",
                        "gt_quality": "",
                        "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
                        # Manual review columns; fill later in results.csv.
                        "manual_fidelity_1_5": "",
                        "manual_detail_1_5": "",
                        "manual_no_hallucination_1_5": "",
                        "manual_text_logo_1_5": "",
                        "manual_color_1_5": "",
                        "manual_usable_yes_no": "",
                        "manual_notes": "",
                    }

                    try:
                        (
                            response,
                            output_path,
                            output_mime,
                            latency,
                            usage,
                            retry_stats,
                        ) = generate(
                            client=client,
                            model_cfg=model_cfg,
                            input_path=input_path,
                            target_size=target,
                            aspect_ratio=assets["aspect_label"],
                            output_stem=output_stem,
                            max_attempts=args.max_attempts,
                            nb2_4k_pre_delay=args.nb2_4k_pre_delay,
                        )

                        generated = open_rgb(output_path)
                        output_width, output_height = generated.size
                        output_dimensions = (
                            f"{output_width}x{output_height}"
                        )

                        gt_path = (
                            source_root
                            / "ground_truth"
                            / model_key
                            / target
                            / f"gt_run_{run_number:02d}_"
                              f"{output_width}x{output_height}.png"
                        )

                        gt_quality = make_ground_truth(
                            normalized_source=normalized,
                            output_width=output_width,
                            output_height=output_height,
                            gt_path=gt_path,
                            mode=args.mode,
                        )

                        costs = estimate_cost(
                            model_cfg,
                            target,
                            usage,
                        )

                        if args.skip_metrics:
                            metrics = {
                                "psnr": math.nan,
                                "ssim": math.nan,
                                "sharpness_laplacian": math.nan,
                                "gt_sharpness_laplacian": math.nan,
                                "sharpness_ratio_vs_gt": math.nan,
                                "rgb_mae": math.nan,
                            }
                        else:
                            metrics = calculate_metrics(
                                output_path,
                                gt_path,
                            )

                        row.update({
                            "status": "ok",
                            "error_class": "",
                            "error": "",
                            "output_path": str(output_path),
                            "output_mime_type": output_mime,
                            "output_dimensions": output_dimensions,
                            "output_file_bytes": (
                                output_path.stat().st_size
                            ),
                            "output_sha256": sha256_file(output_path),
                            "gt_path": str(gt_path),
                            "gt_quality": gt_quality,
                            "latency_sec": latency,
                            **retry_stats,
                            **usage,
                            **costs,
                            **metrics,
                        })

                        print(
                            f"  OK -> {output_dimensions} "
                            f"| final={latency:.2f}s "
                            f"| e2e={retry_stats['end_to_end_latency_sec']:.2f}s "
                            f"| attempts={int(retry_stats['attempts_used'])} "
                            f"| est {money_usd(costs['estimated_total_cost_usd'])} "
                            f"| GT={gt_quality} "
                            f"| SSIM={safe_num(metrics['ssim'], 4)}"
                        )

                    except KeyboardInterrupt:
                        print("\nInterrupted by user. Saving partial results...")
                        raise

                    except Exception as exc:
                        error_class = classify_error(exc)
                        print(
                            f"  ERROR [{error_class}]: {exc}"
                        )

                        if isinstance(exc, GenerateRequestError):
                            failed_attempts = exc.attempts_used
                            failed_retries = max(0, exc.attempts_used - 1)
                            failed_retry_wait = exc.retry_wait_sec
                            failed_pre_delay = exc.pre_request_delay_sec
                            failed_api_total = exc.api_attempts_total_sec
                            failed_e2e = exc.end_to_end_latency_sec
                        else:
                            failed_attempts = 1
                            failed_retries = 0
                            failed_retry_wait = 0.0
                            failed_pre_delay = 0.0
                            failed_api_total = 0.0
                            failed_e2e = 0.0

                        row.update({
                            "status": "error",
                            "error_class": error_class,
                            "error": str(exc),
                            "latency_sec": 0.0,
                            "attempts_used": failed_attempts,
                            "retry_count": failed_retries,
                            "retry_wait_sec": failed_retry_wait,
                            "pre_request_delay_sec": failed_pre_delay,
                            "api_attempts_total_sec": failed_api_total,
                            "end_to_end_latency_sec": failed_e2e,
                            "prompt_tokens": 0,
                            "input_text_tokens": 0,
                            "input_image_tokens": 0,
                            "candidate_tokens_total": 0,
                            "output_text_tokens": 0,
                            "output_image_tokens_reported": 0,
                            "thought_tokens": 0,
                            "cached_tokens": 0,
                            "tool_use_prompt_tokens": 0,
                            "total_tokens": 0,
                            "traffic_type": "",
                            "input_cost_usd": 0.0,
                            "text_thinking_cost_usd": 0.0,
                            "image_output_cost_usd": 0.0,
                            "image_output_tokens_used_for_cost": 0,
                            "image_output_tokens_cost_source": "",
                            "estimated_total_cost_usd": 0.0,
                            "estimated_total_cost_vnd": math.nan,
                            "psnr": math.nan,
                            "ssim": math.nan,
                            "sharpness_laplacian": math.nan,
                            "gt_sharpness_laplacian": math.nan,
                            "sharpness_ratio_vs_gt": math.nan,
                            "rgb_mae": math.nan,
                        })

                    finally:
                        results.append(row)

                    # Keep completed test cases slightly separated. Retry/backoff
                    # is handled inside generate() and tracked independently.
                    if args.request_delay > 0:
                        time.sleep(args.request_delay)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            client.close()
        except Exception:
            pass

    if not results:
        print("Không có result nào để lưu.")
        return 1

    df = pd.DataFrame(results)

    results_path = run_root / "results.csv"
    summary_path = run_root / "summary.csv"
    report_path = run_root / "report.html"

    df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    create_summary_csv(
        df,
        summary_path,
        args.mode,
    )

    create_html_report(
        df,
        report_path,
        run_root,
        manifest,
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Run folder: ", run_root.resolve())
    print("Results CSV:", results_path.resolve())
    print("Summary CSV:", summary_path.resolve())
    print("Native CSV: ", (run_root / "summary_native.csv").resolve())
    print("Reference CSV:", (run_root / "summary_reference.csv").resolve())
    print("Recommendations:", (run_root / "recommendations.csv").resolve())
    print("HTML report:", report_path.resolve())
    print("Manifest:    ", (run_root / "manifest.json").resolve())

    successful = df[df["status"] == "ok"]

    if successful.empty:
        print("Successful requests: 0")
        return 2

    total_cost = float(
        successful["estimated_total_cost_usd"].sum()
    )

    print()
    print(f"Successful requests: {len(successful)}")
    print(f"Estimated API cost:  {money_usd(total_cost)}")

    if USD_TO_VND > 0:
        print(
            f"Approx VND:          "
            f"{total_cost * USD_TO_VND:,.0f} VND"
        )

    print(
        "Billing note: Flex pricing is used only after trafficType=ON_DEMAND_FLEX verification; Google Cloud Billing is authoritative; "
        "the value above is an estimate."
    )

    low_confidence = int(
        (
            successful["gt_quality"]
            == "upscaled_reference"
        ).sum()
    )

    if low_confidence:
        print(
            f"Warning: {low_confidence} successful rows use "
            "upscaled_reference; pixel metrics are lower-confidence."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
