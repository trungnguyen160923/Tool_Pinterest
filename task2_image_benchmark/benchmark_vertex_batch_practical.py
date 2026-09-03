#!/usr/bin/env python3
"""
Gemini Image Upscale Benchmark — BATCH / Practical Mode
Google Cloud / Gemini Enterprise Agent Platform

Purpose
-------
Benchmark the same ordinary source image(s) across the current Gemini image models
using Vertex/Agent Platform Batch Inference, focusing on:

  - image fidelity diagnostics
  - 1K output quality
  - Batch cost per image
  - Batch job turnaround / throughput

Current Batch limitation (important)
------------------------------------
As of 2026-08-26, Gemini Batch Inference supports image output only at the default
1K resolution. 2K and 4K image outputs are NOT supported in Batch.

Therefore the Batch benchmark matrix is intentionally:

  Nano Banana 2 Lite -> 1K
  Nano Banana 2      -> 1K
  Nano Banana Pro    -> 1K

The same normalized source image is sent to every selected model.

Batch pricing snapshot
----------------------
Batch is 50% cheaper than Standard/real-time for these image models:

  Nano Banana 2 Lite:
    input             $0.125 / 1M tokens
    text/thinking     $0.75  / 1M tokens
    image output      $15.00 / 1M image tokens

  Nano Banana 2:
    input             $0.25  / 1M tokens
    text/thinking     $1.50  / 1M tokens
    image output      $30.00 / 1M image tokens

  Nano Banana Pro:
    input             $1.00  / 1M tokens
    text/thinking     $6.00  / 1M tokens
    image output      $60.00 / 1M image tokens

1K output uses 1120 image tokens for all three models.

Authentication
--------------
Uses Google Cloud Application Default Credentials (ADC), same as the Standard
benchmark:

  gcloud auth application-default login
  gcloud auth application-default set-quota-project YOUR_PROJECT_ID

.env
----
Required:
  GOOGLE_CLOUD_PROJECT=gemini-image-benchmark
  GOOGLE_CLOUD_LOCATION=global
  GOOGLE_GENAI_USE_ENTERPRISE=True

Optional:
  USD_TO_VND=26000
  BATCH_GCS_BUCKET=gemini-image-benchmark-gemini-batch-benchmark
  BATCH_GCS_BUCKET_LOCATION=US

Dependencies
------------
  google-genai
  google-cloud-storage
  pillow
  numpy
  pandas
  scikit-image
  opencv-python-headless
  python-dotenv

Notes
-----
- Batch is asynchronous. It can queue under shared-capacity pressure.
- Most jobs complete within 24 hours after they start running; queue time can be
  longer. This script can either wait for completion or submit-only and collect later.
- Batch automatically retries internally; this benchmark does not add per-request
  retry logic like the Standard benchmark.
- Batch jobs are grouped by model: one job for Lite, one for NB2, one for Pro.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import html
import json
import math
import mimetypes
import os
import re
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
from google.genai import types
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden


# ============================================================
# PATHS / ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RUNS_DIR = BASE_DIR / "runs_batch"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

PROJECT_ID = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
LOCATION = (os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip()
USE_ENTERPRISE = (os.getenv("GOOGLE_GENAI_USE_ENTERPRISE") or "").strip().lower()

try:
    USD_TO_VND = float((os.getenv("USD_TO_VND") or "0").strip())
except ValueError:
    USD_TO_VND = 0.0

BATCH_GCS_BUCKET = (
    os.getenv("BATCH_GCS_BUCKET")
    or (f"{PROJECT_ID}-gemini-batch-benchmark" if PROJECT_ID else "")
).strip()

BATCH_GCS_BUCKET_LOCATION = (
    os.getenv("BATCH_GCS_BUCKET_LOCATION") or "US"
).strip()

if not PROJECT_ID:
    raise RuntimeError(
        "Thiếu GOOGLE_CLOUD_PROJECT trong .env.\n"
        "Ví dụ: GOOGLE_CLOUD_PROJECT=gemini-image-benchmark"
    )

if USE_ENTERPRISE not in {"true", "1", "yes"}:
    raise RuntimeError(
        "GOOGLE_GENAI_USE_ENTERPRISE phải = True trong .env."
    )

if not BATCH_GCS_BUCKET:
    raise RuntimeError("Không xác định được BATCH_GCS_BUCKET.")


# ============================================================
# BENCHMARK CONFIG
# ============================================================

PRICING_SNAPSHOT_DATE = "2026-08-26"
BENCHMARK_MODE = "batch_practical"
TARGET = "1K"
OUTPUT_MIME_TYPE = "image/jpeg"
OUTPUT_JPEG_QUALITY = 100
TEMPERATURE = 0.0
POLL_SECONDS_DEFAULT = 30.0
BATCH_PREFIX_ROOT = "gemini-image-batch-benchmark"

# Batch currently supports only default 1K image output.
BATCH_SUPPORTED_TARGETS = ("1K",)

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
Return the edited image. Keep any accompanying text to an absolute minimum.
""".strip()


# ============================================================
# MODEL / BATCH PRICING DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    model_id: str
    batch_input_per_1m: float
    batch_text_thinking_output_per_1m: float
    batch_image_output_per_1m: float
    input_image_tokens: int
    output_image_tokens_1k: int
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
        batch_input_per_1m=0.125,
        batch_text_thinking_output_per_1m=0.75,
        batch_image_output_per_1m=15.0,
        input_image_tokens=1120,
        output_image_tokens_1k=1120,
        aspect_ratios=COMMON_RATIOS,
    ),
    "nb2": ModelConfig(
        key="nb2",
        name="Nano Banana 2",
        model_id="gemini-3.1-flash-image",
        batch_input_per_1m=0.25,
        batch_text_thinking_output_per_1m=1.50,
        batch_image_output_per_1m=30.0,
        input_image_tokens=1120,
        output_image_tokens_1k=1120,
        aspect_ratios=RATIOS_WITH_9_21,
    ),
    "pro": ModelConfig(
        key="pro",
        name="Nano Banana Pro",
        model_id="gemini-3-pro-image",
        batch_input_per_1m=1.00,
        batch_text_thinking_output_per_1m=6.00,
        batch_image_output_per_1m=60.0,
        input_image_tokens=560,
        output_image_tokens_1k=1120,
        aspect_ratios=RATIOS_WITH_9_21,
    ),
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Gemini image upscale using Google Cloud Batch Inference "
            "(current image output limit: 1K only)."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Model cần test. Mặc định: lite nb2 pro.",
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "Chỉ chạy source theo filename/glob. Ví dụ: "
            "--sources img1.jpg 'product-*.jpg'"
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Số request lặp cho mỗi source/model. Mặc định: 1.",
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=POLL_SECONDS_DEFAULT,
        help=f"Chu kỳ poll trạng thái batch. Mặc định {POLL_SECONDS_DEFAULT}s.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ hiển thị plan/cost preview; không upload GCS, không tạo batch job.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Không hỏi xác nhận trước khi submit batch jobs.",
    )

    parser.add_argument(
        "--submit-only",
        action="store_true",
        help=(
            "Upload input + submit jobs rồi thoát. Dùng --collect RUN_FOLDER sau đó "
            "để poll/download/report."
        ),
    )

    parser.add_argument(
        "--collect",
        type=str,
        default=None,
        metavar="RUN_FOLDER",
        help=(
            "Resume một batch run đã submit. Ví dụ: "
            "--collect runs_batch/20260826_210000"
        ),
    )

    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Không tính SSIM/PSNR/Sharpness/MAE.",
    )

    parser.add_argument(
        "--keep-gcs",
        action="store_true",
        help=(
            "Giữ nguyên input/jsonl/output artifacts trên GCS sau khi collect. "
            "Mặc định script cũng KHÔNG xóa để thuận tiện audit; flag này chỉ được "
            "ghi vào manifest cho rõ intent."
        ),
    )

    return parser.parse_args()


# ============================================================
# GENERIC HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def money_usd(value: float) -> str:
    return f"${value:.6f}"


def money_vnd(value_usd: float) -> str:
    if USD_TO_VND <= 0:
        return ""
    return f"{value_usd * USD_TO_VND:,.0f} VND"


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
    return "image/png"


def extension_for_mime(mime: str | None) -> str:
    mime = (mime or "").lower()
    if mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".bin"


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180]


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
    if not patterns:
        return sources

    selected: list[Path] = []
    for source in sources:
        if any(
            source.name == pattern or fnmatch.fnmatch(source.name, pattern)
            for pattern in patterns
        ):
            selected.append(source)
    return selected


def attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


# ============================================================
# IMAGE PREPARATION
# ============================================================

def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


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


def prepare_source(
    source_path: Path,
    source_root: Path,
    allowed_ratios: frozenset[str],
) -> dict[str, Any]:
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
    input_path = inputs_dir / "input_source.png"

    normalized.save(normalized_path, format="PNG", optimize=True)
    normalized.save(input_path, format="PNG", optimize=True)

    return {
        "source_path": source_path,
        "source_name": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_width": source_width,
        "source_height": source_height,
        "normalized": normalized,
        "normalized_path": normalized_path,
        "input_path": input_path,
        "normalized_width": normalized.width,
        "normalized_height": normalized.height,
        "aspect_label": aspect_label,
        "aspect_value": aspect_value,
        "crop_retained_fraction": retained_fraction,
    }


def make_reference(
    normalized_source: Image.Image,
    output_width: int,
    output_height: int,
    reference_path: Path,
) -> str:
    native = (
        normalized_source.width >= output_width
        and normalized_source.height >= output_height
    )

    reference = normalized_source.resize(
        (output_width, output_height),
        Image.Resampling.LANCZOS,
    )

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference.save(reference_path, format="PNG")

    return "native_downscale" if native else "upscaled_reference"


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
    reference_path: Path,
) -> dict[str, float]:
    output = load_rgb_array(output_path)
    ref = load_rgb_array(reference_path)

    if output.shape != ref.shape:
        ref = cv2.resize(
            ref,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    psnr = peak_signal_noise_ratio(ref, output, data_range=255)
    ssim = structural_similarity(
        ref,
        output,
        channel_axis=2,
        data_range=255,
    )

    output_sharpness = laplacian_sharpness(output)
    ref_sharpness = laplacian_sharpness(ref)

    mae = float(
        np.mean(
            np.abs(
                output.astype(np.float32) - ref.astype(np.float32)
            )
        )
    )

    sharpness_ratio = (
        output_sharpness / ref_sharpness
        if ref_sharpness > 0
        else math.nan
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "sharpness_laplacian": output_sharpness,
        "reference_sharpness_laplacian": ref_sharpness,
        "sharpness_ratio_vs_reference": sharpness_ratio,
        "rgb_mae": mae,
    }


# ============================================================
# GOOGLE CLIENTS / GCS
# ============================================================

def build_genai_client() -> genai.Client:
    return genai.Client(
        enterprise=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=types.HttpOptions(api_version="v1"),
    )


def build_storage_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)


def ensure_bucket(storage_client: storage.Client) -> storage.Bucket:
    bucket = storage_client.bucket(BATCH_GCS_BUCKET)

    try:
        bucket.reload()
        return bucket
    except NotFound:
        print(
            f"GCS bucket chưa tồn tại. Đang tạo gs://{BATCH_GCS_BUCKET} "
            f"location={BATCH_GCS_BUCKET_LOCATION}..."
        )
        try:
            bucket = storage_client.create_bucket(
                BATCH_GCS_BUCKET,
                location=BATCH_GCS_BUCKET_LOCATION,
            )
            print("  Created bucket.")
            return bucket
        except Forbidden as exc:
            raise RuntimeError(
                "Không có quyền tạo GCS bucket. Hãy tự tạo bucket rồi đặt "
                "BATCH_GCS_BUCKET=<bucket-name> trong .env."
            ) from exc


def gcs_uri(bucket_name: str, blob_name: str) -> str:
    return f"gs://{bucket_name}/{blob_name}"


def upload_file(
    bucket: storage.Bucket,
    local_path: Path,
    blob_name: str,
    content_type: str | None = None,
) -> str:
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(
        str(local_path),
        content_type=content_type or mimetypes.guess_type(local_path.name)[0],
    )
    return gcs_uri(bucket.name, blob_name)


def download_blob_to(
    blob: storage.Blob,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    rest = uri[5:]
    bucket_name, _, blob_name = rest.partition("/")
    return bucket_name, blob_name


# ============================================================
# BATCH REQUEST BUILDING
# ============================================================

def batch_generation_config(aspect_ratio: str) -> dict[str, Any]:
    # Batch image output is currently limited to 1K. Explicitly requesting 1K keeps
    # the request self-documenting while staying inside the supported Batch feature.
    return {
        "temperature": TEMPERATURE,
        "candidateCount": 1,
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {
            "aspectRatio": aspect_ratio,
            "imageSize": "1K",
            "imageOutputOptions": {
                "mimeType": OUTPUT_MIME_TYPE,
                "compressionQuality": OUTPUT_JPEG_QUALITY,
            },
        },
    }


def make_batch_request_line(
    request_key: str,
    input_gcs_uri: str,
    input_mime: str,
    aspect_ratio: str,
) -> dict[str, Any]:
    return {
        "key": request_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "fileUri": input_gcs_uri,
                                "mimeType": input_mime,
                            }
                        },
                        {"text": PROMPT},
                    ],
                }
            ],
            "generationConfig": batch_generation_config(aspect_ratio),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# JOB STATE / TIMING HELPERS
# ============================================================

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


def normalize_job_state(state: Any) -> str:
    if state is None:
        return "UNKNOWN"

    name = getattr(state, "name", None)
    if name:
        return str(name)

    text = str(state)
    if "." in text:
        text = text.split(".")[-1]
    return text


def datetime_to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def datetime_delta_seconds(start: Any, end: Any) -> float:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max(0.0, (end - start).total_seconds())
    return math.nan


def job_output_directory(job: Any) -> str:
    output_info = attr(job, "output_info", None)
    value = first_non_none(
        attr(output_info, "gcs_output_directory", None),
        attr(output_info, "gcsOutputDirectory", None),
    )
    return str(value or "")


def job_error_text(job: Any) -> str:
    error = attr(job, "error", None)
    if error is None:
        return ""
    return str(error)


# ============================================================
# BATCH OUTPUT PARSING
# ============================================================

def deep_get(obj: Any, *path: str) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def find_request_key(record: dict[str, Any]) -> str:
    candidates = [
        record.get("key"),
        deep_get(record, "instance", "key"),
        deep_get(record, "request", "key"),
        deep_get(record, "prediction", "key"),
    ]
    for value in candidates:
        if value:
            return str(value)
    return ""


def find_response_object(record: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("response", "prediction", "predictions"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value[0]

    # Some outputs can wrap prediction under an object.
    for outer in ("output", "result"):
        value = record.get(outer)
        if isinstance(value, dict):
            for key in ("response", "prediction"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    return nested

    # Defensive fallback if the record itself looks like a GenerateContentResponse.
    if "candidates" in record:
        return record

    return None


def find_error_in_record(record: dict[str, Any]) -> str:
    for key in ("error", "status"):
        value = record.get(key)
        if value:
            if isinstance(value, dict):
                # Ignore success-like empty status objects.
                code = value.get("code")
                message = value.get("message")
                if code or message:
                    return json.dumps(value, ensure_ascii=False)
            elif str(value).strip():
                return str(value)

    response = find_response_object(record)
    if response and response.get("error"):
        return str(response.get("error"))

    return ""


def response_parts(response: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = response.get("candidates") or []
    if not candidates:
        return []

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    return [p for p in parts if isinstance(p, dict)]


def extract_generated_image_from_record(
    record: dict[str, Any],
    output_stem: Path,
    storage_client: storage.Client,
) -> tuple[Path, str, str]:
    response = find_response_object(record)
    if not response:
        raise RuntimeError("Batch output record không chứa response/prediction.")

    text_parts: list[str] = []

    for part in response_parts(response):
        text = part.get("text")
        if text:
            text_parts.append(str(text))

        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            raw = base64.b64decode(inline["data"])
            mime = (
                inline.get("mimeType")
                or inline.get("mime_type")
                or OUTPUT_MIME_TYPE
            )
            ext = extension_for_mime(mime)
            output_path = output_stem.with_suffix(ext)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(raw)

            with Image.open(output_path) as img:
                img.verify()

            return output_path, str(mime), "\n".join(text_parts)

        file_data = part.get("fileData") or part.get("file_data")
        if isinstance(file_data, dict):
            uri = file_data.get("fileUri") or file_data.get("file_uri")
            if uri and str(uri).startswith("gs://"):
                mime = (
                    file_data.get("mimeType")
                    or file_data.get("mime_type")
                    or OUTPUT_MIME_TYPE
                )
                bucket_name, blob_name = parse_gcs_uri(str(uri))
                blob = storage_client.bucket(bucket_name).blob(blob_name)
                output_path = output_stem.with_suffix(extension_for_mime(mime))
                download_blob_to(blob, output_path)

                with Image.open(output_path) as img:
                    img.verify()

                return output_path, str(mime), "\n".join(text_parts)

    raise RuntimeError("Batch response không chứa image output.")


def token_count_for_modality(items: Any, modality_name: str) -> int:
    total = 0
    target = modality_name.upper()

    for item in items or []:
        if not isinstance(item, dict):
            continue

        modality = str(item.get("modality") or "").upper()
        if target not in modality:
            continue

        value = first_non_none(
            item.get("tokenCount"),
            item.get("token_count"),
            item.get("tokens"),
            0,
        )
        total += int(value or 0)

    return total


def extract_usage_from_record(record: dict[str, Any]) -> dict[str, int]:
    response = find_response_object(record) or {}
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}

    prompt_details = (
        usage.get("promptTokensDetails")
        or usage.get("prompt_tokens_details")
        or []
    )
    candidate_details = (
        usage.get("candidatesTokensDetails")
        or usage.get("candidates_tokens_details")
        or []
    )

    return {
        "prompt_tokens": int(
            first_non_none(
                usage.get("promptTokenCount"),
                usage.get("prompt_token_count"),
                0,
            )
            or 0
        ),
        "input_text_tokens": token_count_for_modality(prompt_details, "TEXT"),
        "input_image_tokens": token_count_for_modality(prompt_details, "IMAGE"),
        "candidate_tokens_total": int(
            first_non_none(
                usage.get("candidatesTokenCount"),
                usage.get("candidates_token_count"),
                0,
            )
            or 0
        ),
        "output_text_tokens": token_count_for_modality(candidate_details, "TEXT"),
        "output_image_tokens_reported": token_count_for_modality(
            candidate_details, "IMAGE"
        ),
        "thought_tokens": int(
            first_non_none(
                usage.get("thoughtsTokenCount"),
                usage.get("thoughts_token_count"),
                0,
            )
            or 0
        ),
        "cached_tokens": int(
            first_non_none(
                usage.get("cachedContentTokenCount"),
                usage.get("cached_content_token_count"),
                0,
            )
            or 0
        ),
        "total_tokens": int(
            first_non_none(
                usage.get("totalTokenCount"),
                usage.get("total_token_count"),
                0,
            )
            or 0
        ),
    }


# ============================================================
# COST
# ============================================================

def estimate_batch_cost(
    model_cfg: ModelConfig,
    usage: dict[str, int],
) -> dict[str, Any]:
    # Prefer exact prompt token count from Batch response. If missing, fall back to
    # documented image-input tokens only. This fallback intentionally does NOT invent
    # text input tokens and is therefore a lower-bound estimate.
    if usage["prompt_tokens"] > 0:
        input_tokens_for_cost = usage["prompt_tokens"]
        input_token_source = "usage_metadata"
    else:
        input_tokens_for_cost = model_cfg.input_image_tokens
        input_token_source = "documented_image_only_fallback"

    input_cost = (
        input_tokens_for_cost
        / 1_000_000
        * model_cfg.batch_input_per_1m
    )

    if usage["output_image_tokens_reported"] > 0:
        output_image_tokens = usage["output_image_tokens_reported"]
        output_image_token_source = "usage_metadata"
    else:
        output_image_tokens = model_cfg.output_image_tokens_1k
        output_image_token_source = "documented_1k_fallback"

    image_output_cost = (
        output_image_tokens
        / 1_000_000
        * model_cfg.batch_image_output_per_1m
    )

    text_thinking_tokens = (
        usage["output_text_tokens"] + usage["thought_tokens"]
    )

    text_thinking_cost = (
        text_thinking_tokens
        / 1_000_000
        * model_cfg.batch_text_thinking_output_per_1m
    )

    estimated_total = input_cost + image_output_cost + text_thinking_cost

    return {
        "input_tokens_used_for_cost": input_tokens_for_cost,
        "input_tokens_cost_source": input_token_source,
        "output_image_tokens_used_for_cost": output_image_tokens,
        "output_image_tokens_cost_source": output_image_token_source,
        "input_cost_usd": input_cost,
        "text_thinking_cost_usd": text_thinking_cost,
        "image_output_cost_usd": image_output_cost,
        "estimated_total_cost_usd": estimated_total,
        "estimated_total_cost_vnd": (
            estimated_total * USD_TO_VND
            if USD_TO_VND > 0
            else math.nan
        ),
    }


def minimum_preview_cost(model_cfg: ModelConfig) -> float:
    # Lower-bound preview: documented input-image tokens + fixed 1K output tokens.
    return (
        model_cfg.input_image_tokens
        / 1_000_000
        * model_cfg.batch_input_per_1m
        + model_cfg.output_image_tokens_1k
        / 1_000_000
        * model_cfg.batch_image_output_per_1m
    )


# ============================================================
# PLAN / MANIFEST
# ============================================================

def print_plan(
    sources: list[Path],
    model_keys: list[str],
    runs: int,
) -> None:
    total_requests = len(sources) * len(model_keys) * runs
    preview = sum(
        minimum_preview_cost(MODELS[key])
        for key in model_keys
    ) * len(sources) * runs

    print("=" * 82)
    print("GEMINI IMAGE UPSCALE BENCHMARK — BATCH / PRACTICAL / 1K")
    print("=" * 82)
    print(f"Project:        {PROJECT_ID}")
    print(f"Location:       {LOCATION}")
    print(f"Dataset:        {DATASET_DIR}")
    print(f"GCS bucket:     gs://{BATCH_GCS_BUCKET}")
    print(f"Bucket loc:     {BATCH_GCS_BUCKET_LOCATION}")
    print(f"Source images:  {len(sources)}")
    print(f"Models:         {', '.join(model_keys)}")
    print(f"Runs/test:      {runs}")
    print(f"Total requests: {total_requests}")
    print(f"Target:         1K only (Batch limitation)")
    print(f"Pricing date:   {PRICING_SNAPSHOT_DATE}")
    print()
    print("Minimum Batch cost preview:")
    print(f"  USD: {money_usd(preview)}")
    if USD_TO_VND > 0:
        print(f"  VND: {money_vnd(preview)}")
    print("  + prompt text input + possible text/thinking output.")
    print()
    print("Per-image minimum preview:")
    for key in model_keys:
        cfg = MODELS[key]
        cost = minimum_preview_cost(cfg)
        print(f"  - {cfg.name:<24} 1K ≈ {money_usd(cost)}")
    print()
    print("Important: Batch image output currently does NOT support 2K or 4K.")
    print("=" * 82)


def create_manifest(
    run_root: Path,
    sources: list[Path],
    model_keys: list[str],
    args: argparse.Namespace,
    allowed_ratios: frozenset[str],
    run_id: str,
) -> dict[str, Any]:
    manifest = {
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "backend": "Gemini Enterprise Agent Platform / Batch Inference",
        "api_version": "v1",
        "benchmark_mode": BENCHMARK_MODE,
        "input_strategy": "same normalized source image sent to every selected model",
        "batch_target": "1K",
        "batch_image_output_limit_note": (
            "Batch image output is currently limited to default 1K; 2K and 4K are unsupported."
        ),
        "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
        "usd_to_vnd": USD_TO_VND if USD_TO_VND > 0 else None,
        "batch_pricing_discount_vs_standard": "50%",
        "gcs_bucket": BATCH_GCS_BUCKET,
        "gcs_bucket_location": BATCH_GCS_BUCKET_LOCATION,
        "gcs_prefix": f"{BATCH_PREFIX_ROOT}/{run_id}",
        "poll_seconds": max(1.0, args.poll_seconds),
        "runs_per_test": args.runs,
        "skip_metrics": args.skip_metrics,
        "keep_gcs": bool(args.keep_gcs),
        "temperature": TEMPERATURE,
        "output_mime_type": OUTPUT_MIME_TYPE,
        "output_jpeg_quality": OUTPUT_JPEG_QUALITY,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "allowed_common_aspect_ratios": sorted(allowed_ratios),
        "sources": [
            {
                "name": p.name,
                "path": str(p),
                "sha256": sha256_file(p),
            }
            for p in sources
        ],
        "models": [
            {
                "model_key": key,
                "model_name": MODELS[key].name,
                "model_id": MODELS[key].model_id,
                "batch_input_per_1m": MODELS[key].batch_input_per_1m,
                "batch_text_thinking_output_per_1m": (
                    MODELS[key].batch_text_thinking_output_per_1m
                ),
                "batch_image_output_per_1m": MODELS[key].batch_image_output_per_1m,
                "documented_input_image_tokens": MODELS[key].input_image_tokens,
                "documented_output_image_tokens_1k": (
                    MODELS[key].output_image_tokens_1k
                ),
            }
            for key in model_keys
        ],
    }

    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_root / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")

    return manifest


# ============================================================
# PREPARE + SUBMIT
# ============================================================

def prepare_batch_run(
    run_root: Path,
    run_id: str,
    sources: list[Path],
    model_keys: list[str],
    args: argparse.Namespace,
    allowed_ratios: frozenset[str],
    storage_client: storage.Client,
    bucket: storage.Bucket,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_map: dict[str, Any] = {}
    prepared_assets: dict[str, dict[str, Any]] = {}

    # 1) Prepare one normalized practical input per source and upload once.
    for source_path in sources:
        print()
        print(f"Preparing source: {source_path.name}")
        source_root = run_root / source_path.stem
        source_root.mkdir(parents=True, exist_ok=True)

        assets = prepare_source(source_path, source_root, allowed_ratios)
        prepared_assets[source_path.name] = assets

        retained_pct = assets["crop_retained_fraction"] * 100
        print(
            f"  normalized={assets['normalized_width']}x{assets['normalized_height']} "
            f"aspect={assets['aspect_label']} retained={retained_pct:.2f}%"
        )

        source_tag = sanitize_key(
            f"{source_path.stem}-{assets['source_sha256'][:10]}"
        )
        blob_name = (
            f"{BATCH_PREFIX_ROOT}/{run_id}/inputs/{source_tag}/input_source.png"
        )

        input_gcs_uri = upload_file(
            bucket,
            assets["input_path"],
            blob_name,
            content_type="image/png",
        )
        assets["input_gcs_uri"] = input_gcs_uri
        print(f"  uploaded -> {input_gcs_uri}")

    # 2) Build one JSONL per model. Google recommends combining small requests into
    #    larger jobs; with a fixed model per job, this is the natural grouping.
    model_jsonl_uris: dict[str, str] = {}

    for model_key in model_keys:
        cfg = MODELS[model_key]
        rows: list[dict[str, Any]] = []

        for source_path in sources:
            assets = prepared_assets[source_path.name]
            for run_number in range(1, args.runs + 1):
                request_key = sanitize_key(
                    f"{source_path.stem}__{model_key}__1K__run{run_number:02d}"
                )

                rows.append(
                    make_batch_request_line(
                        request_key=request_key,
                        input_gcs_uri=assets["input_gcs_uri"],
                        input_mime="image/png",
                        aspect_ratio=assets["aspect_label"],
                    )
                )

                request_map[request_key] = {
                    "request_key": request_key,
                    "source": source_path.name,
                    "source_sha256": assets["source_sha256"],
                    "source_width": assets["source_width"],
                    "source_height": assets["source_height"],
                    "normalized_width": assets["normalized_width"],
                    "normalized_height": assets["normalized_height"],
                    "crop_retained_fraction": assets["crop_retained_fraction"],
                    "aspect_ratio": assets["aspect_label"],
                    "model_key": model_key,
                    "model_name": cfg.name,
                    "model_id": cfg.model_id,
                    "target": "1K",
                    "run": run_number,
                    "input_path": str(assets["input_path"]),
                    "input_gcs_uri": assets["input_gcs_uri"],
                }

        local_jsonl = run_root / "batch_requests" / f"{model_key}.jsonl"
        write_jsonl(local_jsonl, rows)

        blob_name = (
            f"{BATCH_PREFIX_ROOT}/{run_id}/requests/{model_key}.jsonl"
        )
        jsonl_uri = upload_file(
            bucket,
            local_jsonl,
            blob_name,
            content_type="application/jsonl",
        )
        model_jsonl_uris[model_key] = jsonl_uri
        print(
            f"Prepared {cfg.name}: {len(rows)} request(s) -> {jsonl_uri}"
        )

    request_map_path = run_root / "request_map.json"
    request_map_path.write_text(
        json.dumps(request_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prep_info = {
        "model_jsonl_uris": model_jsonl_uris,
        "request_count": len(request_map),
    }
    (run_root / "prep_info.json").write_text(
        json.dumps(prep_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return request_map, prep_info


def submit_jobs(
    run_root: Path,
    run_id: str,
    model_keys: list[str],
    prep_info: dict[str, Any],
    client: genai.Client,
) -> dict[str, Any]:
    jobs: dict[str, Any] = {}

    for model_key in model_keys:
        cfg = MODELS[model_key]
        src_uri = prep_info["model_jsonl_uris"][model_key]
        dest_uri = (
            f"gs://{BATCH_GCS_BUCKET}/{BATCH_PREFIX_ROOT}/{run_id}/outputs/{model_key}"
        )
        display_name = sanitize_key(
            f"img-upscale-batch-{model_key}-{run_id}"
        )

        print()
        print(f"Submitting Batch job: {cfg.name}")
        print(f"  src : {src_uri}")
        print(f"  dest: {dest_uri}")

        submitted_at = utc_now_iso()

        job = client.batches.create(
            model=cfg.model_id,
            src=src_uri,
            config=types.CreateBatchJobConfig(
                dest=dest_uri,
                display_name=display_name,
            ),
        )

        state = normalize_job_state(attr(job, "state", None))
        print(f"  job : {job.name}")
        print(f"  state: {state}")

        jobs[model_key] = {
            "model_key": model_key,
            "model_name": cfg.name,
            "model_id": cfg.model_id,
            "job_name": str(job.name),
            "display_name": display_name,
            "src_uri": src_uri,
            "dest_uri": dest_uri,
            "submitted_at_utc": submitted_at,
            "last_state": state,
            "output_directory": job_output_directory(job),
            "job_error": job_error_text(job),
        }

        save_jobs_file(run_root, jobs)

    return jobs


def save_jobs_file(run_root: Path, jobs: dict[str, Any]) -> None:
    (run_root / "batch_jobs.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_jobs_file(run_root: Path) -> dict[str, Any]:
    path = run_root / "batch_jobs.json"
    if not path.exists():
        raise RuntimeError(f"Không tìm thấy {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_request_map(run_root: Path) -> dict[str, Any]:
    path = run_root / "request_map.json"
    if not path.exists():
        raise RuntimeError(f"Không tìm thấy {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# POLL JOBS
# ============================================================

def poll_jobs_until_terminal(
    run_root: Path,
    jobs: dict[str, Any],
    client: genai.Client,
    poll_seconds: float,
) -> dict[str, Any]:
    poll_seconds = max(1.0, poll_seconds)
    pending = set(jobs.keys())

    while pending:
        print()
        print(f"Polling {len(pending)} Batch job(s)... {utc_now_iso()}")

        for model_key in list(pending):
            meta = jobs[model_key]
            job = client.batches.get(name=meta["job_name"])
            state = normalize_job_state(attr(job, "state", None))

            meta["last_state"] = state
            meta["create_time"] = datetime_to_iso(attr(job, "create_time", None))
            meta["start_time"] = datetime_to_iso(attr(job, "start_time", None))
            meta["end_time"] = datetime_to_iso(attr(job, "end_time", None))
            meta["update_time"] = datetime_to_iso(attr(job, "update_time", None))
            meta["output_directory"] = (
                job_output_directory(job) or meta.get("output_directory", "")
            )
            meta["job_error"] = job_error_text(job)

            job_duration = datetime_delta_seconds(
                attr(job, "start_time", None),
                attr(job, "end_time", None),
            )
            turnaround = datetime_delta_seconds(
                attr(job, "create_time", None),
                attr(job, "end_time", None),
            )
            meta["job_processing_sec"] = job_duration
            meta["job_turnaround_sec"] = turnaround

            completion_stats = attr(job, "completion_stats", None)
            if completion_stats is not None:
                meta["completion_stats"] = {
                    "successful_count": int(
                        first_non_none(
                            attr(completion_stats, "successful_count", None),
                            attr(completion_stats, "successfulCount", None),
                            0,
                        )
                        or 0
                    ),
                    "failed_count": int(
                        first_non_none(
                            attr(completion_stats, "failed_count", None),
                            attr(completion_stats, "failedCount", None),
                            0,
                        )
                        or 0
                    ),
                    "incomplete_count": int(
                        first_non_none(
                            attr(completion_stats, "incomplete_count", None),
                            attr(completion_stats, "incompleteCount", None),
                            0,
                        )
                        or 0
                    ),
                }

            print(
                f"  {MODELS[model_key].name:<24} {state:<24} "
                f"output={meta.get('output_directory') or meta.get('dest_uri')}"
            )

            if state in TERMINAL_STATES:
                pending.remove(model_key)
                meta["terminal_observed_at_utc"] = utc_now_iso()

        save_jobs_file(run_root, jobs)

        if pending:
            time.sleep(poll_seconds)

    return jobs


# ============================================================
# DOWNLOAD + PARSE RESULTS
# ============================================================

def list_jsonl_blobs_under_uri(
    storage_client: storage.Client,
    uri: str,
) -> list[storage.Blob]:
    bucket_name, prefix = parse_gcs_uri(uri.rstrip("/"))
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket, prefix=prefix))
    return [blob for blob in blobs if blob.name.lower().endswith(".jsonl")]


def load_jsonl_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSONL {path}:{line_number}: {exc}"
                    ) from exc
                if isinstance(record, dict):
                    records.append(record)
    return records


def collect_model_output_records(
    run_root: Path,
    model_key: str,
    job_meta: dict[str, Any],
    storage_client: storage.Client,
) -> list[dict[str, Any]]:
    # output_info.gcs_output_directory is authoritative when present. If absent,
    # fall back to the destination prefix supplied during creation.
    output_uri = (
        job_meta.get("output_directory")
        or job_meta.get("dest_uri")
        or ""
    )

    if not output_uri:
        raise RuntimeError(f"Không có output GCS URI cho model {model_key}.")

    blobs = list_jsonl_blobs_under_uri(storage_client, output_uri)

    if not blobs and job_meta.get("dest_uri") != output_uri:
        blobs = list_jsonl_blobs_under_uri(
            storage_client,
            job_meta["dest_uri"],
        )

    if not blobs:
        raise RuntimeError(
            f"Không tìm thấy predictions JSONL dưới {output_uri}"
        )

    local_dir = run_root / "batch_raw_outputs" / model_key
    local_paths: list[Path] = []

    for i, blob in enumerate(sorted(blobs, key=lambda b: b.name)):
        local_path = local_dir / f"{i:03d}_{Path(blob.name).name}"
        download_blob_to(blob, local_path)
        local_paths.append(local_path)

    return load_jsonl_records(local_paths)


def create_base_result_row(
    meta: dict[str, Any],
    job_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_mode": BENCHMARK_MODE,
        "source": meta["source"],
        "source_sha256": meta["source_sha256"],
        "source_width": meta["source_width"],
        "source_height": meta["source_height"],
        "normalized_width": meta["normalized_width"],
        "normalized_height": meta["normalized_height"],
        "crop_retained_fraction": meta["crop_retained_fraction"],
        "aspect_ratio": meta["aspect_ratio"],
        "model_key": meta["model_key"],
        "model_name": meta["model_name"],
        "model_id": meta["model_id"],
        "target": "1K",
        "run": meta["run"],
        "request_key": meta["request_key"],
        "input_path": meta["input_path"],
        "input_gcs_uri": meta["input_gcs_uri"],
        "output_path": "",
        "output_mime_type": "",
        "output_dimensions": "",
        "output_file_bytes": 0,
        "output_sha256": "",
        "reference_path": "",
        "reference_quality": "",
        "batch_job_name": job_meta.get("job_name", ""),
        "batch_job_state": job_meta.get("last_state", ""),
        "batch_job_processing_sec": job_meta.get("job_processing_sec", math.nan),
        "batch_job_turnaround_sec": job_meta.get("job_turnaround_sec", math.nan),
        "batch_output_directory": job_meta.get("output_directory", ""),
        "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
        # Manual review columns
        "manual_fidelity_1_5": "",
        "manual_detail_1_5": "",
        "manual_no_hallucination_1_5": "",
        "manual_text_logo_1_5": "",
        "manual_color_1_5": "",
        "manual_usable_yes_no": "",
        "manual_notes": "",
    }


def collect_all_results(
    run_root: Path,
    jobs: dict[str, Any],
    request_map: dict[str, Any],
    storage_client: storage.Client,
    skip_metrics: bool,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    # Load normalized source images once for reference construction.
    normalized_cache: dict[str, Image.Image] = {}
    for meta in request_map.values():
        source = meta["source"]
        if source not in normalized_cache:
            path = run_root / Path(source).stem / "reference" / "reference_normalized.png"
            normalized_cache[source] = open_rgb(path)

    for model_key, job_meta in jobs.items():
        state = job_meta.get("last_state", "")

        if state != "JOB_STATE_SUCCEEDED":
            # Emit an error row for every request belonging to this failed job.
            for request_key, meta in request_map.items():
                if meta["model_key"] != model_key:
                    continue
                row = create_base_result_row(meta, job_meta)
                row.update({
                    "status": "error",
                    "error_class": "batch_job_failed",
                    "error": job_meta.get("job_error") or state,
                })
                results.append(row)
            continue

        try:
            records = collect_model_output_records(
                run_root,
                model_key,
                job_meta,
                storage_client,
            )
        except Exception as exc:
            for request_key, meta in request_map.items():
                if meta["model_key"] != model_key:
                    continue
                row = create_base_result_row(meta, job_meta)
                row.update({
                    "status": "error",
                    "error_class": "batch_output_download",
                    "error": str(exc),
                })
                results.append(row)
            continue

        for record in records:
            request_key = find_request_key(record)

            if not request_key:
                # Save unmapped raw record for audit, but don't invent mapping.
                continue

            if request_key not in request_map:
                continue

            meta = request_map[request_key]
            if meta["model_key"] != model_key:
                continue

            # Defensive duplicate handling: preserve the first mapped response and
            # ignore extra duplicate keys so benchmark accounting stays one-row/request.
            # Raw JSONL remains on disk for billing/audit investigation.
            if request_key in seen_keys:
                continue
            seen_keys.add(request_key)

            row = create_base_result_row(meta, job_meta)
            error_text = find_error_in_record(record)

            if error_text and not find_response_object(record):
                row.update({
                    "status": "error",
                    "error_class": "batch_request_error",
                    "error": error_text,
                })
                results.append(row)
                continue

            try:
                output_stem = (
                    run_root
                    / Path(meta["source"]).stem
                    / "outputs"
                    / model_key
                    / "1K"
                    / f"run_{int(meta['run']):02d}"
                )

                output_path, output_mime, response_text = (
                    extract_generated_image_from_record(
                        record,
                        output_stem,
                        storage_client,
                    )
                )

                output_img = open_rgb(output_path)
                output_width, output_height = output_img.size

                reference_path = (
                    run_root
                    / Path(meta["source"]).stem
                    / "ground_truth"
                    / model_key
                    / "1K"
                    / f"gt_run_{int(meta['run']):02d}_"
                      f"{output_width}x{output_height}.png"
                )

                reference_quality = make_reference(
                    normalized_source=normalized_cache[meta["source"]],
                    output_width=output_width,
                    output_height=output_height,
                    reference_path=reference_path,
                )

                usage = extract_usage_from_record(record)
                costs = estimate_batch_cost(MODELS[model_key], usage)

                if skip_metrics:
                    metrics = {
                        "psnr": math.nan,
                        "ssim": math.nan,
                        "sharpness_laplacian": math.nan,
                        "reference_sharpness_laplacian": math.nan,
                        "sharpness_ratio_vs_reference": math.nan,
                        "rgb_mae": math.nan,
                    }
                else:
                    metrics = calculate_metrics(output_path, reference_path)

                row.update({
                    "status": "ok",
                    "error_class": "",
                    "error": error_text,
                    "response_text": response_text,
                    "output_path": str(output_path),
                    "output_mime_type": output_mime,
                    "output_dimensions": f"{output_width}x{output_height}",
                    "output_file_bytes": output_path.stat().st_size,
                    "output_sha256": sha256_file(output_path),
                    "reference_path": str(reference_path),
                    "reference_quality": reference_quality,
                    **usage,
                    **costs,
                    **metrics,
                })

            except Exception as exc:
                row.update({
                    "status": "error",
                    "error_class": "batch_response_parse",
                    "error": str(exc),
                })

            results.append(row)

    # Emit missing-response rows for submitted requests not present in output.
    existing_keys = {row.get("request_key") for row in results}
    for request_key, meta in request_map.items():
        if request_key in existing_keys:
            continue
        job_meta = jobs.get(meta["model_key"], {})
        row = create_base_result_row(meta, job_meta)
        row.update({
            "status": "error",
            "error_class": "missing_batch_response",
            "error": "Không tìm thấy response tương ứng trong Batch output JSONL.",
        })
        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# SUMMARY / RECOMMENDATION / REPORT
# ============================================================

def create_summary(df: pd.DataFrame) -> pd.DataFrame:
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
                "reference_quality",
                lambda s: int((s == "native_downscale").sum()),
            ),
            upscaled_ref_rows=(
                "reference_quality",
                lambda s: int((s == "upscaled_reference").sum()),
            ),
            avg_batch_job_processing_sec=("batch_job_processing_sec", "mean"),
            avg_batch_job_turnaround_sec=("batch_job_turnaround_sec", "mean"),
            avg_estimated_cost_usd=("estimated_total_cost_usd", "mean"),
            total_estimated_cost_usd=("estimated_total_cost_usd", "sum"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_thought_tokens=("thought_tokens", "mean"),
            avg_output_image_tokens=("output_image_tokens_reported", "mean"),
            avg_ssim=("ssim", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_sharpness=("sharpness_laplacian", "mean"),
            avg_sharpness_ratio_vs_reference=(
                "sharpness_ratio_vs_reference", "mean"
            ),
            avg_rgb_mae=("rgb_mae", "mean"),
        )
        .sort_values("avg_estimated_cost_usd")
    )

    return summary


def create_recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    candidates = summary.copy()
    best_ssim = safe_float(candidates["avg_ssim"].max())

    if math.isnan(best_ssim):
        eligible = candidates
        metric_basis = "cost only; quality metrics unavailable"
        confidence = "provisional"
    else:
        eligible = candidates[candidates["avg_ssim"] >= best_ssim - 0.01]
        metric_basis = "SSIM screening + Batch cost"
        confidence = (
            "higher"
            if int(candidates["native_rows"].sum()) > 0
            else "provisional"
        )

    pick = eligible.sort_values("avg_estimated_cost_usd").iloc[0]

    return pd.DataFrame([
        {
            "target": "1K",
            "recommended_model": pick["model_name"],
            "model_key": pick["model_key"],
            "price_per_image_usd": pick["avg_estimated_cost_usd"],
            "avg_ssim": pick["avg_ssim"],
            "avg_psnr": pick["avg_psnr"],
            "avg_rgb_mae": pick["avg_rgb_mae"],
            "confidence": confidence,
            "metric_basis": metric_basis,
            "selection_rule": (
                "Lowest Batch cost among models within 0.01 SSIM of best average SSIM."
            ),
        }
    ])


def relative_uri(path_value: Any, run_root: Path) -> str:
    if not path_value:
        return ""
    try:
        return os.path.relpath(
            str(path_value), str(run_root)
        ).replace("\\", "/")
    except Exception:
        return ""


def image_cell(path_value: Any, run_root: Path, width: int = 180) -> str:
    if not path_value:
        return ""
    uri = html.escape(relative_uri(path_value, run_root))
    return f'<a href="{uri}" target="_blank"><img src="{uri}" width="{width}"></a>'


def create_html_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    recommendation: pd.DataFrame,
    run_root: Path,
    manifest: dict[str, Any],
    jobs: dict[str, Any],
) -> None:
    ok = df[df["status"] == "ok"].copy()
    total_cost = (
        float(ok["estimated_total_cost_usd"].sum()) if not ok.empty else 0.0
    )

    job_rows = []
    for key, meta in jobs.items():
        stats = meta.get("completion_stats") or {}
        job_rows.append(
            "<tr>"
            f"<td>{html.escape(MODELS[key].name)}</td>"
            f"<td>{html.escape(str(meta.get('last_state', '')))}</td>"
            f"<td>{safe_num(meta.get('job_processing_sec'), 2)}s</td>"
            f"<td>{safe_num(meta.get('job_turnaround_sec'), 2)}s</td>"
            f"<td>{int(stats.get('successful_count', 0))}</td>"
            f"<td>{int(stats.get('failed_count', 0))}</td>"
            f"<td>{html.escape(str(meta.get('output_directory') or meta.get('dest_uri') or ''))}</td>"
            "</tr>"
        )

    summary_rows = []
    for _, row in summary.iterrows():
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_name']))}</td>"
            f"<td>{int(row['requests'])}</td>"
            f"<td>{int(row['native_rows'])}</td>"
            f"<td>{int(row['upscaled_ref_rows'])}</td>"
            f"<td>${safe_num(row['avg_estimated_cost_usd'], 6)}</td>"
            f"<td>{safe_num(row['avg_batch_job_processing_sec'], 2)}s</td>"
            f"<td>{safe_num(row['avg_batch_job_turnaround_sec'], 2)}s</td>"
            f"<td>{safe_num(row['avg_ssim'], 4)}</td>"
            f"<td>{safe_num(row['avg_psnr'], 2)}</td>"
            f"<td>{safe_num(row['avg_rgb_mae'], 2)}</td>"
            "</tr>"
        )

    rec_html = "<p>No recommendation available.</p>"
    if not recommendation.empty:
        r = recommendation.iloc[0]
        rec_html = (
            "<div class='recommend'>"
            f"<strong>Automatic 1K Batch value pick:</strong> "
            f"{html.escape(str(r['recommended_model']))}<br>"
            f"Estimated price/image: ${safe_num(r['price_per_image_usd'], 6)}<br>"
            f"Confidence: {html.escape(str(r['confidence']))}<br>"
            f"Rule: {html.escape(str(r['selection_rule']))}"
            "</div>"
        )

    detail_rows = []
    for _, row in df.iterrows():
        if row.get("status") == "ok":
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['source']))}</td>"
                f"<td>{html.escape(str(row['model_name']))}</td>"
                f"<td>1K</td>"
                f"<td>{image_cell(row['input_path'], run_root)}</td>"
                f"<td>{image_cell(row['output_path'], run_root)}</td>"
                f"<td>{image_cell(row['reference_path'], run_root)}</td>"
                f"<td>{html.escape(str(row['output_dimensions']))}</td>"
                f"<td>{html.escape(str(row['reference_quality']))}</td>"
                f"<td>${safe_num(row['estimated_total_cost_usd'], 6)}</td>"
                f"<td>{safe_num(row['prompt_tokens'], 0)}</td>"
                f"<td>{safe_num(row['thought_tokens'], 0)}</td>"
                f"<td>{safe_num(row['output_image_tokens_reported'], 0)}</td>"
                f"<td>{safe_num(row['ssim'], 4)}</td>"
                f"<td>{safe_num(row['psnr'], 2)}</td>"
                f"<td>{safe_num(row['sharpness_ratio_vs_reference'], 3)}</td>"
                f"<td>{safe_num(row['rgb_mae'], 2)}</td>"
                "</tr>"
            )
        else:
            detail_rows.append(
                "<tr class='error'>"
                f"<td>{html.escape(str(row.get('source', '')))}</td>"
                f"<td>{html.escape(str(row.get('model_name', '')))}</td>"
                "<td>1K</td>"
                f"<td colspan='13'>{html.escape(str(row.get('error', '')))}</td>"
                "</tr>"
            )

    total_vnd = (
        f"{total_cost * USD_TO_VND:,.0f} VND"
        if USD_TO_VND > 0
        else "N/A (set USD_TO_VND in .env)"
    )

    report_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemini Image Upscale Benchmark — Batch</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; white-space: nowrap; }}
    th {{ background: #f0f2f5; position: sticky; top: 0; }}
    .note {{ background: #f5f7fb; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
    .warning {{ background: #fff4e5; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
    .cost {{ background: #eef9f0; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
    .recommend {{ background: #eef3ff; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
    .table-wrap {{ overflow-x: auto; }}
    .error {{ background: #ffecec; }}
    img {{ max-width: 180px; height: auto; display: block; }}
    code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Gemini Image Upscale Benchmark — BATCH</h1>

  <div class="note">
    <strong>Backend:</strong> Gemini Enterprise Agent Platform / Batch Inference<br>
    <strong>Project:</strong> {html.escape(PROJECT_ID)}<br>
    <strong>Location:</strong> {html.escape(LOCATION)}<br>
    <strong>Mode:</strong> practical<br>
    <strong>Input strategy:</strong> same normalized source sent to each model<br>
    <strong>Target:</strong> 1K only<br>
    <strong>Pricing snapshot:</strong> {PRICING_SNAPSHOT_DATE}<br>
    <strong>Batch discount:</strong> 50% vs Standard/real-time pricing
  </div>

  <div class="warning">
    <strong>Batch limitation:</strong> Gemini Batch Inference currently supports image
    output only at the default 1K resolution. 2K and 4K image outputs are not supported.
    Batch is asynchronous and should not be compared to Standard using per-request latency.
    Use job turnaround/throughput instead.
  </div>

  <div class="cost">
    <strong>Estimated successful-request cost:</strong> ${total_cost:.6f} USD<br>
    <strong>Approx VND:</strong> {total_vnd}<br>
    <small>Estimate from usage metadata + Batch pricing snapshot. Google Cloud Billing is authoritative.</small>
  </div>

  {rec_html}

  <h2>Batch jobs</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Model</th><th>State</th><th>Processing</th><th>Turnaround</th><th>Success</th><th>Failed</th><th>Output GCS</th></tr></thead>
    <tbody>{''.join(job_rows)}</tbody>
  </table></div>

  <h2>Aggregate summary — 1K Batch</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Model</th><th>Requests</th><th>Native</th><th>Upscaled-ref</th><th>Avg cost/image</th><th>Job processing</th><th>Job turnaround</th><th>Avg SSIM</th><th>Avg PSNR</th><th>Avg MAE</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table></div>

  <div class="warning">
    SSIM/PSNR/MAE are fidelity diagnostics. When reference quality is
    <code>upscaled_reference</code>, manual review for identity, hallucination,
    text/logo, color and detail remains required.
  </div>

  <h2>Detailed results</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Source</th><th>Model</th><th>Target</th><th>Input</th><th>Output</th><th>Reference</th><th>Output size</th><th>Reference quality</th><th>Cost</th><th>Prompt tokens</th><th>Thinking tokens</th><th>Image tokens</th><th>SSIM</th><th>PSNR</th><th>Sharp/ref</th><th>MAE</th></tr></thead>
    <tbody>{''.join(detail_rows)}</tbody>
  </table></div>
</body>
</html>
"""

    (run_root / "report.html").write_text(report_html, encoding="utf-8")


# ============================================================
# FINALIZE RUN
# ============================================================

def finalize_run(
    run_root: Path,
    manifest: dict[str, Any],
    jobs: dict[str, Any],
    request_map: dict[str, Any],
    storage_client: storage.Client,
    skip_metrics: bool,
) -> int:
    print()
    print("Collecting Batch outputs...")

    df = collect_all_results(
        run_root=run_root,
        jobs=jobs,
        request_map=request_map,
        storage_client=storage_client,
        skip_metrics=skip_metrics,
    )

    results_path = run_root / "results.csv"
    summary_path = run_root / "summary.csv"
    recommendation_path = run_root / "recommendations.csv"

    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    summary = create_summary(df)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    recommendation = create_recommendation(summary)
    recommendation.to_csv(
        recommendation_path,
        index=False,
        encoding="utf-8-sig",
    )

    create_html_report(
        df=df,
        summary=summary,
        recommendation=recommendation,
        run_root=run_root,
        manifest=manifest,
        jobs=jobs,
    )

    ok = df[df["status"] == "ok"]
    errors_df = df[df["status"] != "ok"]

    print()
    print("=" * 82)
    print("BATCH BENCHMARK DONE")
    print("=" * 82)
    print("Run folder:      ", run_root.resolve())
    print("Results CSV:     ", results_path.resolve())
    print("Summary CSV:     ", summary_path.resolve())
    print("Recommendations: ", recommendation_path.resolve())
    print("HTML report:     ", (run_root / "report.html").resolve())
    print("Batch jobs:      ", (run_root / "batch_jobs.json").resolve())
    print("Manifest:        ", (run_root / "manifest.json").resolve())
    print()
    print(f"Successful requests: {len(ok)}")
    print(f"Failed/missing:       {len(errors_df)}")

    if not ok.empty:
        total_cost = float(ok["estimated_total_cost_usd"].sum())
        print(f"Estimated Batch cost: {money_usd(total_cost)}")
        if USD_TO_VND > 0:
            print(f"Approx VND:           {money_vnd(total_cost)}")

    if not recommendation.empty:
        rec = recommendation.iloc[0]
        print()
        print(
            "Automatic 1K Batch value pick: "
            f"{rec['recommended_model']} "
            f"({money_usd(float(rec['price_per_image_usd']))}/image)"
        )

    print()
    print(
        "Billing note: Batch pricing estimate only. Google Cloud Billing is authoritative."
    )
    print(
        "Batch note: 2K/4K image output is currently unsupported; compare Batch vs Standard at 1K."
    )

    return 0 if not ok.empty else 2


# ============================================================
# COLLECT EXISTING RUN
# ============================================================

def resolve_collect_run(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        candidate = BASE_DIR / path
        if candidate.exists():
            path = candidate
        else:
            candidate = RUNS_DIR / path_value
            if candidate.exists():
                path = candidate

    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"RUN_FOLDER không tồn tại: {path}")
    return path


def collect_existing_run(args: argparse.Namespace) -> int:
    run_root = resolve_collect_run(args.collect)

    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Không tìm thấy manifest.json trong {run_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = load_jobs_file(run_root)
    request_map = load_request_map(run_root)

    client = build_genai_client()
    storage_client = build_storage_client()

    try:
        jobs = poll_jobs_until_terminal(
            run_root,
            jobs,
            client,
            poll_seconds=args.poll_seconds,
        )
        return finalize_run(
            run_root,
            manifest,
            jobs,
            request_map,
            storage_client,
            skip_metrics=args.skip_metrics,
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    if args.collect:
        return collect_existing_run(args)

    if args.runs < 1:
        raise ValueError("--runs phải >= 1.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    sources = filter_sources(discover_sources(), args.sources)
    if not sources:
        raise RuntimeError(
            f"Không tìm thấy source phù hợp trong {DATASET_DIR}."
        )

    model_keys = args.models
    allowed_ratios = ratio_intersection(model_keys)

    print_plan(sources, model_keys, args.runs)

    if args.dry_run:
        print("DRY RUN: chưa tạo bucket, chưa upload GCS, chưa submit Batch job.")
        return 0

    if not args.yes:
        answer = input(
            "Submit Batch jobs có tính phí và dùng Cloud Storage? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Đã hủy. Chưa submit Batch job.")
            return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest(
        run_root=run_root,
        sources=sources,
        model_keys=model_keys,
        args=args,
        allowed_ratios=allowed_ratios,
        run_id=run_id,
    )

    storage_client = build_storage_client()
    bucket = ensure_bucket(storage_client)
    client = build_genai_client()

    try:
        request_map, prep_info = prepare_batch_run(
            run_root=run_root,
            run_id=run_id,
            sources=sources,
            model_keys=model_keys,
            args=args,
            allowed_ratios=allowed_ratios,
            storage_client=storage_client,
            bucket=bucket,
        )

        jobs = submit_jobs(
            run_root=run_root,
            run_id=run_id,
            model_keys=model_keys,
            prep_info=prep_info,
            client=client,
        )

        print()
        print("Batch jobs submitted successfully.")
        print("Run folder:", run_root.resolve())

        if args.submit_only:
            print()
            print("SUBMIT-ONLY: script sẽ thoát và không chờ job hoàn tất.")
            print("Collect sau bằng:")
            print(
                f'  python "{Path(__file__).name}" '
                f'--collect "{run_root}"'
            )
            return 0

        jobs = poll_jobs_until_terminal(
            run_root,
            jobs,
            client,
            poll_seconds=args.poll_seconds,
        )

        return finalize_run(
            run_root=run_root,
            manifest=manifest,
            jobs=jobs,
            request_map=request_map,
            storage_client=storage_client,
            skip_metrics=args.skip_metrics,
        )

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
