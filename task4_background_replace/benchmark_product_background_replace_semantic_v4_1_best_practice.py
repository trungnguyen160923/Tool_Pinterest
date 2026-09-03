#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TASK 4 V4.1 — PRODUCT-AWARE BACKGROUND + SEMANTIC PLACEMENT + PHOTOREAL INTEGRATION
============================================================

Purpose
-------
General-purpose product background replacement WITHOUT hardcoding product category.

The pipeline automatically:
  1) analyzes WHAT the product is and HOW it should physically exist
  2) segments the exact product locally
  3) auto-generates a product-aware background brief when no background is supplied
  4) generates a new background
  5) analyzes the generated background to find a physically plausible support surface
  6) selects a placement strategy from semantic analysis
  7) places the original product deterministically with OpenCV/Pillow
  8) harmonizes the product with local background lighting
  9) adds support-aware contact shadow
 10) optionally runs a final image-model relight/blend pass for maximum photorealism
 11) outputs a 4K PNG final image

The deterministic composite is always saved for audit. By default, the final
image-model integration pass is accepted with product-drift warnings rather
than rejected, because photoreal product insertion often requires relighting
and rerendering material edges.

Default best-practice run
-------------------------
  Flex + Nano Banana Pro
  Product-aware auto background
  Final photoreal integration enabled

Benchmark matrix can still be requested with:
  --modes standard flex --models nb2 pro

Google Cloud backend
--------------------
Gemini Enterprise Agent Platform

Default semantic-analysis model
-------------------------------
gemini-2.5-flash

Expected folder
---------------
task4_background_replace/
├─ .env
├─ benchmark_product_background_replace_semantic_v4_1_best_practice.py
└─ dataset/
   ├─ main.webp
   └─ background.txt  (optional; default uses auto background)

.env
----
GOOGLE_CLOUD_PROJECT=gemini-image-benchmark
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_ENTERPRISE=True

Install
-------
pip install -U google-genai pillow pandas python-dotenv numpy opencv-python-headless rembg onnxruntime

Examples
--------
python benchmark_product_background_replace_semantic_v4_1_best_practice.py --dry-run

python benchmark_product_background_replace_semantic_v4_1_best_practice.py

python benchmark_product_background_replace_semantic_v4_1_best_practice.py --modes standard --models pro

python benchmark_product_background_replace_semantic_v4_1_best_practice.py --modes standard flex --models nb2 pro

python benchmark_product_background_replace_semantic_v4_1_best_practice.py ^
  --background "A warm modern living room with natural daylight and neutral beige tones"

Optional manual mask
--------------------
python benchmark_product_background_replace_semantic_v4_1_best_practice.py ^
  --mask-file ".\\dataset\\product_mask.png"

Important
---------
The system does NOT use rules like:
    if product == "rug": ...

Instead, the vision-analysis model returns structured semantics such as:
    physical_form
    natural_support
    natural_orientation
    perspective_transform_allowed
    must_contact_surface
    recommended_camera
    placement_strategy

The local placement engine executes that semantic plan.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False


# ============================================================
# 0. ENV / PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)
else:
    load_dotenv(override=False)

DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs_task4_semantic_v4_1"


# ============================================================
# 1. CONSTANTS
# ============================================================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

IMAGE_MODELS: Dict[str, Dict[str, str]] = {
    "nb2": {
        "label": "Nano Banana 2",
        "id": "gemini-3.1-flash-image",
    },
    "pro": {
        "label": "Nano Banana Pro",
        "id": "gemini-3-pro-image",
    },
}

DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"

# 4K image-output tokens used by current Agent Platform image-model pricing.
OUTPUT_IMAGE_TOKENS_4K = {
    "nb2": 2520,
    "pro": 2000,
}

# Google Cloud / Agent Platform image-generation pricing snapshot.
# USD / 1M tokens.
IMAGE_PRICING = {
    "standard": {
        "nb2": {"input": 0.50, "text_output": 3.00, "image_output": 60.0},
        "pro": {"input": 2.00, "text_output": 12.00, "image_output": 120.0},
    },
    "flex": {
        "nb2": {"input": 0.25, "text_output": 1.50, "image_output": 30.0},
        "pro": {"input": 1.00, "text_output": 6.00, "image_output": 60.0},
    },
}

# Semantic analysis is run on Standard by default.
# Gemini 2.5 Flash Standard pricing:
# input (text/image/video) = $0.30/M
# text output/reasoning    = $2.50/M
ANALYSIS_PRICING = {
    "input": 0.30,
    "output": 2.50,
}

STANDARD_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Server-Timeout": "1800",
}

FLEX_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
    "X-Server-Timeout": "1800",
}

EXPECTED_FLEX_TRAFFIC_TYPE = "ON_DEMAND_FLEX"

MAX_ATTEMPTS = 6
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
RETRY_BACKOFF = {
    "standard": [5, 10, 20, 40, 60],
    "flex": [10, 20, 40, 80, 120],
}

INLINE_MAX_BYTES = 6_500_000


# ============================================================
# 2. DATA CLASSES
# ============================================================

@dataclass
class AnalysisUsage:
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    latency_sec: Optional[float] = None


@dataclass
class ResultRow:
    mode: str
    model_key: str
    model_label: str
    model_id: str

    product_file: str
    source_width: int
    source_height: int

    background_width: Optional[int]
    background_height: Optional[int]

    semantic_product_category: str
    semantic_physical_form: str
    semantic_natural_support: str
    semantic_natural_orientation: str
    semantic_placement_strategy: str
    semantic_confidence: float

    segmentation_method: str
    mask_coverage_ratio: float

    placement_strategy_used: str
    placement_confidence: float
    placement_support_surface: str
    placement_plan_json: str

    generated_background_file: str
    deterministic_output_file: str
    integration_output_file: str
    final_output_file: str

    pipeline_status: str
    integration_status: str
    final_source: str
    status: str
    traffic_type: str
    integration_traffic_type: str
    flex_verified: Optional[bool]
    integration_flex_verified: Optional[bool]
    pricing_verified: bool

    attempts_used: int
    retry_count: int
    retry_wait_sec: float
    generation_latency_sec: Optional[float]
    background_analysis_latency_sec: Optional[float]
    final_integration_latency_sec: Optional[float]

    prompt_tokens: Optional[int]
    thinking_tokens: Optional[int]
    output_image_tokens: Optional[int]
    integration_prompt_tokens: Optional[int]
    integration_thinking_tokens: Optional[int]
    integration_output_image_tokens: Optional[int]
    product_preservation_mae: Optional[float]
    integration_rgb_delta: Optional[float]

    image_generation_cost_usd: Optional[float]
    background_analysis_cost_usd: Optional[float]
    final_integration_cost_usd: Optional[float]
    semantic_analysis_shared_cost_usd: Optional[float]
    estimated_total_pipeline_cost_usd: Optional[float]

    warning_message: str
    error_message: str


# ============================================================
# 3. GENERIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def discover_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    paths = [p for p in folder.iterdir() if is_image(p)]
    paths = [p for p in paths if "mask" not in p.stem.lower()]
    return sorted(paths, key=lambda p: p.name.lower())


def choose_one(paths: List[Path], title: str) -> Path:
    print()
    print(title)
    print("-" * len(title))
    for i, p in enumerate(paths, 1):
        print(f"[{i}] {p.name}")

    while True:
        raw = input("Choose number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(paths):
                return paths[idx - 1]
        print("Invalid choice.")


def resolve_product_file(dataset_dir: Path, explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not is_image(p):
            raise FileNotFoundError(f"Invalid product image: {p}")
        return p

    candidates = discover_images(dataset_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No product image found directly in:\n{dataset_dir}\n\n"
            "Put one product image in dataset/ or pass --product-file."
        )

    if len(candidates) == 1:
        print(f"Auto-selected product image: {candidates[0].name}")
        return candidates[0]

    return choose_one(candidates, "Multiple product images found. Choose ONE:")


def resolve_mask_file(explicit: str) -> Optional[Path]:
    if not explicit.strip():
        return None
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Mask file not found: {p}")
    return p


def resolve_background_override(args) -> Optional[str]:
    if args.background.strip():
        return args.background.strip()

    if args.background_file.strip():
        p = Path(args.background_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Background text file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Background text file is empty: {p}")
        return text

    return None


def resolve_dataset_background_description(args) -> Optional[str]:
    p = Path(args.dataset_dir).expanduser().resolve() / "background.txt"
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        if text:
            print(f"Loaded background description from: {p}")
            return text

    return None


def prompt_background_description() -> str:
    print()
    print("Enter new background description:")
    text = input("Background: ").strip()
    if not text:
        raise ValueError("Background description cannot be empty.")
    return text



def auto_background_description(semantic: Dict[str, Any]) -> str:
    category = str(semantic.get("product_category", "product")).strip() or "product"
    support = str(semantic.get("natural_support", "appropriate support surface")).strip()
    orientation = str(semantic.get("natural_orientation", "natural orientation")).strip()
    camera = str(semantic.get("recommended_camera", "")).strip()
    requirements = semantic.get("support_requirements") or []
    forbidden = semantic.get("forbidden_placements") or []
    functional = normalize_functional_context(semantic.get("functional_context"))

    req_text = "; ".join(str(x).strip() for x in requirements if str(x).strip())
    forbidden_text = "; ".join(str(x).strip() for x in forbidden if str(x).strip())

    lines = [
        f"A premium photorealistic ecommerce lifestyle scene naturally appropriate for a {category}.",
        f"The scene must provide a clean, physically correct {support.lower()} support area for the product.",
        f"The product will be inserted later in its natural {orientation.lower()} orientation, so the camera should make that support plane believable.",
        "Choose an environment where this product would normally be used by a real customer.",
        "Use coherent natural or studio-soft lighting, realistic material texture, and enough negative space around the future product.",
        "Avoid visual clutter and avoid any competing product, duplicate product-like object, text, logo, watermark, or decorative item inside the future placement zone.",
        "The reserved placement zone should look like normal untouched surface, not an empty rectangle or prepared patch.",
    ]

    anchor_type = functional["primary_anchor_type"]
    priority = functional["context_priority"]
    if anchor_type.upper() != "NONE" and priority >= 0.35:
        lines.extend([
            f"FUNCTIONAL ANCHOR: include a clearly visible, realistic {anchor_type} because it determines where the product naturally belongs.",
            f"The future product should be able to sit {functional['anchor_relation'].lower().replace('_', ' ')} that anchor, with {functional['preferred_alignment'].lower().replace('_', ' ')} alignment and {functional['preferred_distance'].lower().replace('_', ' ')} spacing.",
            "Compose the scene so the functional anchor and its adjacent placement area are both visible and usable; do not leave an arbitrary empty area far away from the anchor.",
            "The anchor must remain part of the background, but it must not overlap or cover the future product placement zone.",
        ])
        if functional["scale_reference"].upper() != "NONE":
            lines.append(f"Use this contextual scale reference when composing the scene: {functional['scale_reference']}.")

    if camera:
        lines.append(f"Recommended camera guidance: {camera}.")
    if req_text:
        lines.append(f"Support requirements: {req_text}.")
    if forbidden_text:
        lines.append(f"Forbidden placements/items: {forbidden_text}.")

    return "\n".join(lines)

def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def money(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        return f"${float(v):.6f}"
    except Exception:
        return ""


def number(v: Optional[float], digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return ""


def nearest_aspect_ratio(width: int, height: int) -> str:
    r = width / height
    choices = {
        "1:1": 1.0,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "3:4": 3 / 4,
        "4:3": 4 / 3,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
    }
    return min(choices.items(), key=lambda kv: abs(kv[1] - r))[0]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def clean_token_value(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if math.isnan(float(v)):
            return None
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return None


# ============================================================
# 4. IMAGE ENCODING
# ============================================================

def compress_for_inline(img: Image.Image, max_bytes: int = INLINE_MAX_BYTES) -> Tuple[bytes, str]:
    img = img.convert("RGB")

    last = b""
    for q in [95, 92, 88, 84, 80, 76, 72, 68]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        last = data
        if len(data) <= max_bytes:
            return data, "image/jpeg"

    working = img
    while len(last) > max_bytes:
        w, h = working.size
        nw = max(1, int(w * 0.90))
        nh = max(1, int(h * 0.90))
        if nw == w and nh == h:
            break

        working = working.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        working.save(buf, "JPEG", quality=80, optimize=True)
        last = buf.getvalue()

    return last, "image/jpeg"


def image_part(img: Image.Image, max_bytes: int = INLINE_MAX_BYTES) -> types.Part:
    data, mime = compress_for_inline(img, max_bytes=max_bytes)
    return types.Part.from_bytes(data=data, mime_type=mime)


def image_part_for_vision_analysis(
    img: Image.Image,
    max_side: int = 2048,
    max_bytes: int = 3_500_000,
) -> types.Part:
    """
    Vision-analysis calls need scene geometry and semantics, not full 4K pixels.
    Downsampling keeps requests smaller and avoids transient transport disconnects.
    """
    working = img.convert("RGB")
    w, h = working.size
    longest = max(w, h)

    if longest > max_side:
        scale = max_side / float(longest)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        working = working.resize(size, Image.LANCZOS)

    return image_part(working, max_bytes=max_bytes)


# ============================================================
# 5. GOOGLE CLIENTS
# ============================================================

def create_client(project: str, location: str, mode: str):
    headers = FLEX_HEADERS if mode == "flex" else STANDARD_HEADERS
    return genai.Client(
        enterprise=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=1_800_000,
            headers=headers,
        ),
    )


# ============================================================
# 6. RESPONSE / USAGE HELPERS
# ============================================================

def obj_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    fn = getattr(obj, "model_dump", None)
    if callable(fn):
        try:
            return fn(exclude_none=True)
        except Exception:
            try:
                return fn()
            except Exception:
                pass

    fn = getattr(obj, "to_dict", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass

    return {}


def get_any(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]

    return None


def deep_find_first(obj: Any, keys: set) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
            found = deep_find_first(v, keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_first(item, keys)
            if found is not None:
                return found

    return None


def normalize_traffic_type(value: Any) -> str:
    if value is None:
        return ""

    enum_value = getattr(value, "value", None)
    text = str(enum_value if enum_value else value).strip()

    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text.upper()


def extract_traffic_type(response: Any) -> Tuple[str, str]:
    d = obj_to_dict(response)
    raw = deep_find_first(d, {"trafficType", "traffic_type"})
    raw_text = str(raw) if raw is not None else ""
    return raw_text, normalize_traffic_type(raw)


def extract_usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        d = obj_to_dict(response)
        usage = d.get("usageMetadata") or d.get("usage_metadata") or {}

    prompt = clean_token_value(get_any(usage, ["prompt_token_count", "promptTokenCount"]))
    candidates = clean_token_value(get_any(usage, ["candidates_token_count", "candidatesTokenCount"]))
    thoughts = clean_token_value(get_any(usage, ["thoughts_token_count", "thoughtsTokenCount"]))
    total = clean_token_value(get_any(usage, ["total_token_count", "totalTokenCount"]))

    return {
        "prompt_tokens": prompt,
        "candidate_tokens": candidates,
        "thinking_tokens": thoughts,
        "total_tokens": total,
    }


def extract_image_output_tokens(response: Any) -> Optional[int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return extract_usage(response).get("candidate_tokens")

    details = get_any(usage, ["candidates_tokens_details", "candidatesTokensDetails"])

    if details is not None:
        try:
            total = 0
            found = False
            for item in list(details):
                modality = get_any(item, ["modality"])
                token_count = clean_token_value(get_any(item, ["token_count", "tokenCount"]))
                if modality is not None and token_count is not None and "IMAGE" in str(modality).upper():
                    total += token_count
                    found = True
            if found:
                return total
        except Exception:
            pass

    return extract_usage(response).get("candidate_tokens")


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    chunks: List[str] = []

    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)

    return "\n".join(chunks).strip()


def extract_image_bytes(response: Any) -> Tuple[Optional[bytes], str]:
    candidates = getattr(response, "candidates", None) or []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue

            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or "image/jpeg"

            if data:
                if isinstance(data, str):
                    return base64.b64decode(data), mime
                return bytes(data), mime

    d = obj_to_dict(response)

    def walk(obj):
        if isinstance(obj, dict):
            inline = obj.get("inlineData") or obj.get("inline_data")

            if isinstance(inline, dict):
                data = inline.get("data")
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/jpeg"

                if data:
                    if isinstance(data, str):
                        try:
                            return base64.b64decode(data), mime
                        except Exception:
                            pass
                    elif isinstance(data, (bytes, bytearray)):
                        return bytes(data), mime

            for value in obj.values():
                found = walk(value)
                if found[0] is not None:
                    return found

        elif isinstance(obj, list):
            for item in obj:
                found = walk(item)
                if found[0] is not None:
                    return found

        return None, ""

    return walk(d)


def http_code_from_exception(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)

        if isinstance(value, int):
            return value

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            try:
                return int(enum_value)
            except Exception:
                pass

    text = str(exc)

    for code in RETRYABLE_HTTP_CODES:
        if str(code) in text:
            return code

    return None


def is_retryable_api_exception(exc: Exception) -> bool:
    code = http_code_from_exception(exc)
    if code in RETRYABLE_HTTP_CODES:
        return True

    text = f"{type(exc).__name__}: {exc}".lower()
    retryable_fragments = (
        "remoteprotocolerror",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "readerror",
        "writeerror",
        "network",
    )
    return any(fragment in text for fragment in retryable_fragments)


# ============================================================
# 7. JSON PARSING
# ============================================================

def parse_json_relaxed(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if not text:
        raise ValueError("Empty JSON response.")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ============================================================
# 8. SEMANTIC PRODUCT ANALYSIS
# ============================================================


def semantic_analysis_prompt() -> str:
    return r"""
Analyze the main sellable product in the provided image.

Do NOT decide based on a fixed catalog. Infer the product from visual evidence
and ordinary real-world use.

Your job is to describe:
1. what kind of physical object it is,
2. how it is naturally supported,
3. its natural orientation,
4. whether perspective warping is physically appropriate,
5. whether it must visibly contact a support surface,
6. which deterministic compositing strategy should be used,
7. the functional/contextual anchor that normally determines WHERE it belongs.

Return ONLY valid JSON with exactly this structure:

{
  "product_category": "short natural-language category",
  "physical_form": "PLANAR_FLEXIBLE | PLANAR_RIGID | RIGID_3D | SOFT_3D | HANGING | UNKNOWN",
  "natural_support": "FLOOR | WALL | TABLETOP | SHELF | GROUND | HAND | HANGING | FREESTANDING | UNKNOWN",
  "natural_orientation": "HORIZONTAL | VERTICAL | UPRIGHT | HANGING | VARIABLE | UNKNOWN",
  "placement_strategy": "PLANAR_SURFACE | GROUNDED_OBJECT | SURFACE_OBJECT | WALL_MOUNTED | HANGING_OBJECT | GENERIC",
  "perspective_transform_allowed": true,
  "must_contact_surface": true,
  "recommended_camera": "short description",
  "support_requirements": [
    "short requirement"
  ],
  "forbidden_placements": [
    "short forbidden placement"
  ],
  "functional_context": {
    "primary_anchor_type": "short real-world anchor type or NONE",
    "anchor_relation": "IN_FRONT_OF | BELOW | ABOVE | ON | INSIDE | NEXT_TO | ATTACHED_TO | CENTERED_ON | NEAR | NONE",
    "preferred_alignment": "PARALLEL | PERPENDICULAR | CENTERED | EDGE_ALIGNED | UPRIGHT | NATURAL | NONE",
    "preferred_distance": "TOUCHING | VERY_CLOSE | CLOSE | MODERATE | FLEXIBLE",
    "scale_reference": "short real-world scale reference or NONE",
    "context_priority": 0.0
  },
  "preserve_viewing_angle": true,
  "confidence": 0.0
}

Decision guidance:
- PLANAR_SURFACE means a flat object can be placed using a planar homography on its natural support plane.
- GROUNDED_OBJECT means a 3D object should keep its original viewing angle and be bottom-anchored to floor/ground.
- SURFACE_OBJECT means a 3D object should keep its original viewing angle and sit on a tabletop/shelf/other support surface.
- WALL_MOUNTED means a planar or framed item naturally mounted on a wall.
- HANGING_OBJECT means physically suspended/hanging.
- GENERIC is only for genuinely uncertain cases.
- functional_context is generic and MUST describe the real-world relationship that makes a placement useful, not merely physically possible.
- primary_anchor_type should identify the scene element that normally determines the product's location, such as an entrance threshold, bed, sofa, sink, monitor, wall feature, shelf edge, dining table, person, vehicle, or NONE.
- Example reasoning only: an item intended for an entrance may be anchored to an entrance/threshold; an item used beside furniture may be anchored to that furniture. Do not hardcode categories.
- context_priority: 0 means no reliable functional anchor; 1 means the anchor relationship is essential to a believable placement.
- Infer ordinary functional use when it is clear, but do not invent brand-specific features or product details not supported by the image/category.

Confidence and context_priority must each be between 0 and 1.
""".strip()

def call_json_vision(
    client: Any,
    model_id: str,
    image: Image.Image,
    prompt: str,
    temperature: float = 0.1,
) -> Tuple[Dict[str, Any], AnalysisUsage]:

    content = types.Content(
        role="user",
        parts=[
            image_part_for_vision_analysis(image),
            types.Part.from_text(text=prompt),
        ]
    )

    start = time.perf_counter()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[content],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS or not is_retryable_api_exception(exc):
                raise

            wait = RETRY_BACKOFF["standard"][min(attempt - 1, len(RETRY_BACKOFF["standard"]) - 1)]
            code = http_code_from_exception(exc)
            label = f"HTTP {code}" if code is not None else type(exc).__name__
            print(f"  Retryable vision-analysis {label}; wait {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError("Vision analysis failed after retries.") from last_error

    latency = time.perf_counter() - start
    data = parse_json_relaxed(extract_response_text(response))
    usage = extract_usage(response)

    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("candidate_tokens")

    cost = None
    if prompt_tokens is not None and output_tokens is not None:
        cost = (
            prompt_tokens / 1_000_000 * ANALYSIS_PRICING["input"]
            + output_tokens / 1_000_000 * ANALYSIS_PRICING["output"]
        )

    return data, AnalysisUsage(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=usage.get("total_tokens"),
        estimated_cost_usd=cost,
        latency_sec=latency,
    )



def normalize_functional_context(raw: Any) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    relation = str(raw.get("anchor_relation", "NONE")).strip().upper()
    if relation not in {
        "IN_FRONT_OF", "BELOW", "ABOVE", "ON", "INSIDE",
        "NEXT_TO", "ATTACHED_TO", "CENTERED_ON", "NEAR", "NONE"
    }:
        relation = "NONE"

    alignment = str(raw.get("preferred_alignment", "NONE")).strip().upper()
    if alignment not in {
        "PARALLEL", "PERPENDICULAR", "CENTERED",
        "EDGE_ALIGNED", "UPRIGHT", "NATURAL", "NONE"
    }:
        alignment = "NONE"

    distance = str(raw.get("preferred_distance", "FLEXIBLE")).strip().upper()
    if distance not in {"TOUCHING", "VERY_CLOSE", "CLOSE", "MODERATE", "FLEXIBLE"}:
        distance = "FLEXIBLE"

    try:
        priority = clamp01(float(raw.get("context_priority", 0.0)))
    except Exception:
        priority = 0.0

    return {
        "primary_anchor_type": str(raw.get("primary_anchor_type", "NONE")).strip() or "NONE",
        "anchor_relation": relation,
        "preferred_alignment": alignment,
        "preferred_distance": distance,
        "scale_reference": str(raw.get("scale_reference", "NONE")).strip() or "NONE",
        "context_priority": priority,
    }


def normalize_semantic_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    valid_forms = {
        "PLANAR_FLEXIBLE", "PLANAR_RIGID", "RIGID_3D",
        "SOFT_3D", "HANGING", "UNKNOWN"
    }

    valid_supports = {
        "FLOOR", "WALL", "TABLETOP", "SHELF",
        "GROUND", "HAND", "HANGING", "FREESTANDING", "UNKNOWN"
    }

    valid_orientations = {
        "HORIZONTAL", "VERTICAL", "UPRIGHT",
        "HANGING", "VARIABLE", "UNKNOWN"
    }

    valid_strategies = {
        "PLANAR_SURFACE", "GROUNDED_OBJECT", "SURFACE_OBJECT",
        "WALL_MOUNTED", "HANGING_OBJECT", "GENERIC"
    }

    form = str(raw.get("physical_form", "UNKNOWN")).upper()
    support = str(raw.get("natural_support", "UNKNOWN")).upper()
    orientation = str(raw.get("natural_orientation", "UNKNOWN")).upper()
    strategy = str(raw.get("placement_strategy", "GENERIC")).upper()

    if form not in valid_forms:
        form = "UNKNOWN"
    if support not in valid_supports:
        support = "UNKNOWN"
    if orientation not in valid_orientations:
        orientation = "UNKNOWN"
    if strategy not in valid_strategies:
        strategy = "GENERIC"

    try:
        confidence = clamp01(float(raw.get("confidence", 0.0)))
    except Exception:
        confidence = 0.0

    return {
        "product_category": str(raw.get("product_category", "unknown product")).strip(),
        "physical_form": form,
        "natural_support": support,
        "natural_orientation": orientation,
        "placement_strategy": strategy,
        "perspective_transform_allowed": bool(raw.get("perspective_transform_allowed", False)),
        "must_contact_surface": bool(raw.get("must_contact_surface", True)),
        "recommended_camera": str(raw.get("recommended_camera", "")).strip(),
        "support_requirements": list(raw.get("support_requirements", []) or []),
        "forbidden_placements": list(raw.get("forbidden_placements", []) or []),
        "functional_context": normalize_functional_context(raw.get("functional_context")),
        "preserve_viewing_angle": bool(raw.get("preserve_viewing_angle", True)),
        "confidence": confidence,
    }

def mask_from_explicit_file(mask_path: Path, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as img:
        if img.mode in ("RGBA", "LA"):
            mask = img.getchannel("A")
        else:
            mask = img.convert("L")

        if mask.size != size:
            mask = mask.resize(size, Image.LANCZOS)

        return np.array(mask, dtype=np.uint8)


def alpha_mask_from_product(path: Path) -> Optional[np.ndarray]:
    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA"):
            alpha = np.array(img.getchannel("A"), dtype=np.uint8)

            if alpha.max() > 0 and alpha.min() < 255:
                return alpha

    return None


def rembg_mask(img: Image.Image) -> Optional[np.ndarray]:
    if not REMBG_AVAILABLE:
        return None

    try:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out = rembg_remove(buf.getvalue(), only_mask=True)
        with Image.open(io.BytesIO(out)) as mask:
            return np.array(mask.convert("L"), dtype=np.uint8)
    except Exception:
        return None


def grabcut_mask(img: Image.Image) -> Optional[np.ndarray]:
    try:
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        rect = (
            max(1, int(w * 0.02)),
            max(1, int(h * 0.02)),
            max(2, int(w * 0.96)),
            max(2, int(h * 0.96)),
        )

        cv2.grabCut(
            bgr,
            mask,
            rect,
            bgd,
            fgd,
            7,
            cv2.GC_INIT_WITH_RECT,
        )

        return np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except Exception:
        return None


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    if n <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_label] = 255
    return out


def _mask_cleanup_pixels(
    size: Tuple[int, int],
    contract_ratio: float,
    feather_ratio: float,
) -> Tuple[int, float]:
    """Convert resolution-independent cleanup ratios to stable pixel values."""
    w, h = size
    base = max(1, min(int(w), int(h)))

    contract_ratio = max(0.0, min(float(contract_ratio), 0.01))
    feather_ratio = max(0.0, min(float(feather_ratio), 0.01))

    contract_px = int(round(base * contract_ratio))
    feather_sigma = max(0.0, base * feather_ratio)

    # Keep cleanup deliberately conservative. We want to remove segmentation halo,
    # not eat meaningful product pixels.
    contract_px = min(contract_px, 6)
    feather_sigma = min(feather_sigma, 3.0)
    return contract_px, feather_sigma


def refine_mask(
    mask: np.ndarray,
    contract_ratio: float = 0.0015,
    feather_ratio: float = 0.0008,
) -> np.ndarray:
    """
    Clean a segmentation mask for deterministic ecommerce compositing.

    Best-practice goals:
      - remove isolated background specks,
      - close small holes,
      - keep only the main connected product,
      - contract the alpha edge by a tiny resolution-independent amount to
        suppress source-background fringe,
      - feather only slightly so the product still looks crisp.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if mask.max() <= 1:
        mask = mask * 255

    _, binary = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)
    binary = keep_largest_component(binary)

    contract_px, feather_sigma = _mask_cleanup_pixels(
        (binary.shape[1], binary.shape[0]),
        contract_ratio=contract_ratio,
        feather_ratio=feather_ratio,
    )

    if contract_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contract_px * 2 + 1, contract_px * 2 + 1),
        )
        binary = cv2.erode(binary, k, iterations=1)

    if feather_sigma > 0.0:
        soft = cv2.GaussianBlur(binary, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    else:
        soft = binary

    return soft.astype(np.uint8)


def mask_coverage(mask: np.ndarray) -> float:
    return float((mask > 16).mean())


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 16)

    if len(xs) == 0:
        raise ValueError("Empty product mask.")

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_is_plausible(mask: np.ndarray) -> bool:
    try:
        ratio = mask_coverage(mask)
        bbox_from_mask(mask)
        return 0.005 <= ratio <= 0.95
    except Exception:
        return False


def segment_product(
    product_path: Path,
    img: Image.Image,
    explicit_mask: Optional[Path],
    mask_contract_ratio: float = 0.0015,
    mask_feather_ratio: float = 0.0008,
) -> Tuple[np.ndarray, str]:

    def clean(candidate: np.ndarray) -> np.ndarray:
        return refine_mask(
            candidate,
            contract_ratio=mask_contract_ratio,
            feather_ratio=mask_feather_ratio,
        )

    if explicit_mask is not None:
        mask = clean(mask_from_explicit_file(explicit_mask, img.size))

        if not mask_is_plausible(mask):
            raise ValueError("Explicit mask is not plausible.")

        return mask, "explicit_mask"

    alpha = alpha_mask_from_product(product_path)
    if alpha is not None:
        mask = clean(alpha)

        if mask_is_plausible(mask):
            return mask, "alpha_channel"

    mask = rembg_mask(img)
    if mask is not None:
        mask = clean(mask)

        if mask_is_plausible(mask):
            return mask, "rembg"

    mask = grabcut_mask(img)
    if mask is not None:
        mask = clean(mask)

        if mask_is_plausible(mask):
            return mask, "grabcut"

    raise RuntimeError(
        "Automatic product segmentation failed. "
        "Install rembg/onnxruntime or pass --mask-file."
    )


def rgba_from_product_and_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Attach the segmentation mask as the alpha channel of the original product."""
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L"))
    return out


def crop_rgba_to_mask(
    rgba: Image.Image,
    mask: np.ndarray,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Crop the exact product cutout to the non-empty alpha/mask bounding box."""
    bbox = bbox_from_mask(mask)
    return rgba.crop(bbox), bbox


# ============================================================
# 10. SOURCE PLANAR QUAD
# ============================================================

def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return four 2D points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 quad points, got {pts.shape[0]}.")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # Degenerate protection: if point ordering collapsed because of unusual geometry,
    # fall back to angular ordering around the centroid and rotate to top-left first.
    if len(np.unique(ordered, axis=0)) != 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        cyc = pts[np.argsort(angles)]
        start = int(np.argmin(cyc.sum(axis=1)))
        cyc = np.roll(cyc, -start, axis=0)
        # Ensure TL -> TR -> BR -> BL rather than the opposite winding.
        cross = np.cross(cyc[1] - cyc[0], cyc[2] - cyc[1])
        if cross < 0:
            cyc = cyc[[0, 3, 2, 1]]
        ordered = cyc.astype(np.float32)

    return ordered


def estimate_source_quad(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Estimate the visible planar product corners from the segmentation contour.

    Returned coordinates are relative to the cropped product bbox and ordered
    TL, TR, BR, BL. If no stable quadrilateral can be estimated, the crop
    rectangle is used as a safe fallback.
    """
    x1, y1, x2, y2 = bbox
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    binary = (np.asarray(mask, dtype=np.uint8) > 64).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fallback = np.array(
        [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
        dtype=np.float32,
    )

    if not contours:
        return fallback

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1.0:
        return fallback

    peri = cv2.arcLength(contour, True)
    for eps_factor in (0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            pts[:, 0] -= float(x1)
            pts[:, 1] -= float(y1)
            ordered = order_quad_points(pts)
            if abs(cv2.contourArea(ordered.reshape(-1, 1, 2))) > 1.0:
                return ordered

    # Robust fallback for a planar product whose contour has more than 4 points.
    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect).astype(np.float32)
    pts[:, 0] -= float(x1)
    pts[:, 1] -= float(y1)
    ordered = order_quad_points(pts)

    if abs(cv2.contourArea(ordered.reshape(-1, 1, 2))) <= 1.0:
        return fallback

    return ordered


# ============================================================
# 11. BACKGROUND PROMPT
# ============================================================


def build_background_prompt(
    description: str,
    semantic: Dict[str, Any],
) -> str:

    semantic_json = json.dumps(semantic, ensure_ascii=False, indent=2)
    functional = normalize_functional_context(semantic.get("functional_context"))
    functional_json = json.dumps(functional, ensure_ascii=False, indent=2)

    return f"""
Create ONE photorealistic ecommerce/lifestyle BACKGROUND ONLY.

The foreground product will be added later by deterministic local compositing.
Do not generate or imitate the product itself.

Requested environment:
{description}

The product has already been visually analyzed. Use this semantic information to
design a physically plausible environment and camera viewpoint:

{semantic_json}

Functional placement context:
{functional_json}

Background requirements:
- Create a support surface appropriate to the product's natural_support.
- Respect the recommended_camera.
- Leave a clean, unobstructed placement area on the correct support surface.
- The placement area must be large enough for the hero product.
- The future product must not appear to float.
- If must_contact_surface is true, make the supporting plane clearly visible.
- Avoid objects that would intersect or cover the future product.
- No random text, letters, logos, signs, watermarks, or fake products.
- Do not place another object resembling the hero product in the reserved area.
- Lighting must be coherent and suitable for realistic later compositing.

CRITICAL FUNCTIONAL-PLACEMENT RULE:
- A merely empty support surface is NOT enough when functional_context.context_priority is meaningful.
- If primary_anchor_type is not NONE, include that anchor naturally and visibly in the scene.
- Reserve the future product location in the correct real-world relationship to that anchor.
- The placement zone must be adjacent to / aligned with the anchor as specified by anchor_relation,
  preferred_alignment, and preferred_distance.
- Do NOT create the main empty placement zone far away from the functional anchor.
- Make the anchor geometry visually legible (for example a threshold/edge/centerline when relevant)
  so a later placement-analysis pass can locate and align to it.
- Keep the anchor itself unobstructed enough to understand the product's functional relationship.

CRITICAL SEAMLESS-SURFACE RULE:
- The reserved placement zone must be visually indistinguishable from the surrounding support surface.
- The support surface must continue naturally and seamlessly through the entire placement zone.
- Do NOT create a rectangular patch, platform, mat-shaped region, pedestal, brightness block,
  spotlight rectangle, different flooring material, visible border, tonal box, or artificial empty
  rectangle around the future product location.
- Do NOT "prepare" a visibly different patch for the product. The empty zone should look like normal
  untouched floor/wall/table surface before compositing.
- Surface texture, plank/tile lines, grain, shadows, reflections, and lighting gradients must continue
  through the reserved zone without discontinuity.

Composition best practices:
- Prefer a camera view that makes both the support plane and functional anchor easy to understand geometrically.
- Avoid extreme wide-angle or fisheye distortion.
- Avoid camera angles that make the support plane or anchor relationship ambiguous.
- Keep enough negative space around the future hero product for a natural ecommerce composition.
- Premium commercial ecommerce/lifestyle photography.
- Image only.
""".strip()


def placement_analysis_prompt(
    semantic: Dict[str, Any],
    product_geometry: Optional[Dict[str, Any]] = None,
) -> str:
    semantic_json = json.dumps(semantic, ensure_ascii=False, indent=2)
    geometry_json = json.dumps(product_geometry or {}, ensure_ascii=False, indent=2)

    return f"""
Analyze this GENERATED BACKGROUND image and choose the BEST REAL-WORLD placement
for a foreground product that will be composited later.

Do not merely find an empty support surface. First determine how the product is
functionally used, locate the relevant scene anchor, then choose a placement
relative to that anchor.

Product semantics:
{semantic_json}

Measured source-product geometry:
{geometry_json}

Coordinates must be NORMALIZED to [0,1], where:
(0,0) = top-left
(1,1) = bottom-right

Return ONLY valid JSON:

{{
  "placement_strategy": "PLANAR_SURFACE | GROUNDED_OBJECT | SURFACE_OBJECT | WALL_MOUNTED | HANGING_OBJECT | GENERIC",
  "support_surface_type": "short text",
  "placement_confidence": 0.0,
  "context_score": 0.0,

  "functional_anchor": {{
    "anchor_type": "short detected anchor type or NONE",
    "anchor_bbox": [0.0, 0.0, 0.0, 0.0],
    "anchor_line": [
      [0.0, 0.0],
      [0.0, 0.0]
    ],
    "anchor_point": [0.0, 0.0],
    "relation": "IN_FRONT_OF | BELOW | ABOVE | ON | INSIDE | NEXT_TO | ATTACHED_TO | CENTERED_ON | NEAR | NONE",
    "placement_side": "UP | DOWN | LEFT | RIGHT | OVERLAP | NONE",
    "preferred_alignment": "PARALLEL | PERPENDICULAR | CENTERED | EDGE_ALIGNED | UPRIGHT | NATURAL | NONE",
    "confidence": 0.0
  }},

  "product_quad": [
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]
  ],

  "product_bbox": [0.0, 0.0, 0.0, 0.0],

  "contact_line": [
    [0.0, 0.0],
    [0.0, 0.0]
  ],

  "surface_perspective": {{
    "plane_type": "FLOOR | WALL | TABLETOP | OTHER",
    "farther_image_direction": "UP | DOWN | LEFT | RIGHT | NONE",
    "perspective_strength": 0.0,
    "near_far_scale_ratio": 1.0,
    "confidence": 0.0
  }},

  "rationale": "short explanation of functional anchor + physical placement"
}}

FUNCTIONAL PLACEMENT RULES:
- Product semantics may contain functional_context. Treat that as a high-priority real-world constraint when its context_priority is high.
- Locate the primary_anchor_type in THIS background. Example anchors can be a threshold, door, bed, sofa, sink, shelf edge, monitor, wall feature, table, vehicle, person, etc. Do not assume one category.
- anchor_bbox should bound the visible anchor if it has meaningful area.
- anchor_line should mark the most placement-relevant edge/axis when one exists (for example a threshold edge, shelf edge, tabletop edge, bed edge, wall centerline).
- anchor_point should mark a useful center/reference point.
- placement_side is the image-space side where the product should lie relative to the anchor.
- Prefer the functionally correct anchor-relative location over a visually emptier but functionally wrong location.
- If the product normally belongs close to an anchor, do not place it deep in unrelated empty floor/wall/table space.
- If preferred_alignment is PARALLEL or PERPENDICULAR, align the product's dominant axis accordingly.
- If preferred_alignment is CENTERED, center the product on the relevant anchor axis.
- context_score measures how well the proposed product placement satisfies the real-world anchor relationship, from 0 to 1.
- If the expected anchor is genuinely absent, use anchor_type=NONE and reduce context_score and placement_confidence rather than inventing an anchor.

GEOMETRY RULES:
- product_quad order MUST be top-left, top-right, bottom-right, bottom-left.
- product_quad is REQUIRED and meaningful for PLANAR_SURFACE and WALL_MOUNTED.
- product_bbox = [x1,y1,x2,y2] is REQUIRED for GROUNDED_OBJECT, SURFACE_OBJECT,
  HANGING_OBJECT and GENERIC.
- For GROUNDED_OBJECT, the bottom of product_bbox must physically touch a visible floor/ground/support plane.
- For SURFACE_OBJECT, bottom of product_bbox must sit on a visible tabletop/shelf/support.
- For WALL_MOUNTED, quad must lie on a visible wall plane.
- For PLANAR_SURFACE, quad must lie ON the natural support plane and show realistic perspective for that plane.
- Preserve the product's measured geometry and semantic natural_orientation.
- Do NOT turn a naturally HORIZONTAL product into a portrait/square-looking footprint.
- Do NOT turn a naturally VERTICAL product into a landscape/square-looking footprint.
- Perspective may foreshorten the object, but must not destroy its dominant orientation.
- Keep the product large enough to read as the ecommerce hero, while still physically plausible in the scene.
- Never place the product floating in empty space.
- Never intersect major furniture unless natural occlusion is explicitly appropriate.

SURFACE-PERSPECTIVE RULES:
- Analyze the ACTUAL visible support plane in the generated background, not a generic scene assumption.
- farther_image_direction describes the direction in image space in which points on that plane recede farther from the camera.
- perspective_strength: 0.0 = essentially orthographic/top-down, 1.0 = very strong perspective.
- near_far_scale_ratio describes the expected apparent scale ratio between equal-length segments at the near side and far side of the proposed product footprint.
- For a floor plane whose farther direction is UP, the near/bottom product edge should normally be at least as wide as the far/top edge unless the camera is effectively top-down.
- Do not return a flat rectangular footprint if the floor clearly has noticeable depth perspective.
- Do not return an exaggerated trapezoid if the camera is near-overhead.
- placement_confidence, context_score, functional_anchor.confidence, and surface_perspective.confidence must each be between 0 and 1.
""".strip()

def normalize_point(p: Any) -> List[float]:
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return [0.0, 0.0]

    try:
        return [clamp01(float(p[0])), clamp01(float(p[1]))]
    except Exception:
        return [0.0, 0.0]


def normalize_quad(raw: Any) -> List[List[float]]:
    if not isinstance(raw, list) or len(raw) != 4:
        return []
    return [normalize_point(p) for p in raw]


def normalize_bbox(raw: Any) -> List[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return []

    try:
        x1, y1, x2, y2 = [clamp01(float(v)) for v in raw]
    except Exception:
        return []

    if x2 <= x1 or y2 <= y1:
        return []

    return [x1, y1, x2, y2]


def normalize_surface_perspective(raw: Any, strategy: str) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    plane_type = str(raw.get("plane_type", "OTHER")).strip().upper()
    if plane_type not in {"FLOOR", "WALL", "TABLETOP", "OTHER"}:
        plane_type = "OTHER"

    farther = str(raw.get("farther_image_direction", "NONE")).strip().upper()
    if farther not in {"UP", "DOWN", "LEFT", "RIGHT", "NONE"}:
        farther = "NONE"

    try:
        strength = clamp01(float(raw.get("perspective_strength", 0.0)))
    except Exception:
        strength = 0.0

    try:
        ratio = float(raw.get("near_far_scale_ratio", 1.0))
    except Exception:
        ratio = 1.0
    ratio = max(1.0, min(ratio, 1.60))

    try:
        confidence = clamp01(float(raw.get("confidence", 0.0)))
    except Exception:
        confidence = 0.0

    # Conservative generic fallback. We do not invent strong perspective when the model
    # did not recognize the support plane.
    if strategy == "PLANAR_SURFACE" and plane_type == "OTHER":
        confidence = min(confidence, 0.25)

    return {
        "plane_type": plane_type,
        "farther_image_direction": farther,
        "perspective_strength": strength,
        "near_far_scale_ratio": ratio,
        "confidence": confidence,
    }



def fallback_plan_for_strategy(strategy: str) -> Dict[str, Any]:
    """Generic safety fallback. Intentionally not product-category-specific."""
    empty_anchor = {
        "anchor_type": "NONE",
        "anchor_bbox": [],
        "anchor_line": [],
        "anchor_point": [],
        "relation": "NONE",
        "placement_side": "NONE",
        "preferred_alignment": "NONE",
        "confidence": 0.0,
    }

    if strategy == "PLANAR_SURFACE":
        return {
            "placement_strategy": strategy,
            "support_surface_type": "visible support plane",
            "placement_confidence": 0.20,
            "context_score": 0.0,
            "functional_anchor": empty_anchor,
            "product_quad": [
                [0.20, 0.58],
                [0.80, 0.58],
                [0.88, 0.84],
                [0.12, 0.84],
            ],
            "product_bbox": [],
            "contact_line": [[0.12, 0.84], [0.88, 0.84]],
            "surface_perspective": {
                "plane_type": "OTHER",
                "farther_image_direction": "UP",
                "perspective_strength": 0.25,
                "near_far_scale_ratio": 1.08,
                "confidence": 0.20,
            },
            "rationale": "generic planar fallback",
        }

    if strategy == "WALL_MOUNTED":
        return {
            "placement_strategy": strategy,
            "support_surface_type": "wall",
            "placement_confidence": 0.20,
            "context_score": 0.0,
            "functional_anchor": empty_anchor,
            "product_quad": [
                [0.25, 0.18],
                [0.75, 0.18],
                [0.75, 0.68],
                [0.25, 0.68],
            ],
            "product_bbox": [],
            "contact_line": [],
            "surface_perspective": {
                "plane_type": "WALL",
                "farther_image_direction": "NONE",
                "perspective_strength": 0.0,
                "near_far_scale_ratio": 1.0,
                "confidence": 0.20,
            },
            "rationale": "generic wall fallback",
        }

    return {
        "placement_strategy": strategy if strategy else "GENERIC",
        "support_surface_type": "visible support surface",
        "placement_confidence": 0.20,
        "context_score": 0.0,
        "functional_anchor": empty_anchor,
        "product_quad": [],
        "product_bbox": [0.24, 0.26, 0.76, 0.84],
        "contact_line": [[0.24, 0.84], [0.76, 0.84]],
        "surface_perspective": {
            "plane_type": "OTHER",
            "farther_image_direction": "NONE",
            "perspective_strength": 0.0,
            "near_far_scale_ratio": 1.0,
            "confidence": 0.0,
        },
        "rationale": "generic object fallback",
    }

def normalize_functional_anchor(raw: Any) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    bbox = normalize_bbox(raw.get("anchor_bbox"))
    line = raw.get("anchor_line", [])
    if isinstance(line, list) and len(line) == 2:
        line = [normalize_point(line[0]), normalize_point(line[1])]
    else:
        line = []

    point = raw.get("anchor_point", [])
    if isinstance(point, (list, tuple)) and len(point) == 2:
        point = normalize_point(point)
    else:
        point = []

    relation = str(raw.get("relation", "NONE")).strip().upper()
    if relation not in {
        "IN_FRONT_OF", "BELOW", "ABOVE", "ON", "INSIDE",
        "NEXT_TO", "ATTACHED_TO", "CENTERED_ON", "NEAR", "NONE"
    }:
        relation = "NONE"

    side = str(raw.get("placement_side", "NONE")).strip().upper()
    if side not in {"UP", "DOWN", "LEFT", "RIGHT", "OVERLAP", "NONE"}:
        side = "NONE"

    alignment = str(raw.get("preferred_alignment", "NONE")).strip().upper()
    if alignment not in {
        "PARALLEL", "PERPENDICULAR", "CENTERED",
        "EDGE_ALIGNED", "UPRIGHT", "NATURAL", "NONE"
    }:
        alignment = "NONE"

    try:
        confidence = clamp01(float(raw.get("confidence", 0.0)))
    except Exception:
        confidence = 0.0

    return {
        "anchor_type": str(raw.get("anchor_type", "NONE")).strip() or "NONE",
        "anchor_bbox": bbox,
        "anchor_line": line,
        "anchor_point": point,
        "relation": relation,
        "placement_side": side,
        "preferred_alignment": alignment,
        "confidence": confidence,
    }



def aggregate_analysis_usage(usages: Sequence[AnalysisUsage]) -> AnalysisUsage:
    usages = [u for u in usages if u is not None]
    if not usages:
        return AnalysisUsage()

    def sum_optional(attr: str) -> Optional[float]:
        vals = [getattr(u, attr) for u in usages if getattr(u, attr) is not None]
        return sum(vals) if vals else None

    return AnalysisUsage(
        prompt_tokens=int(sum_optional("prompt_tokens")) if sum_optional("prompt_tokens") is not None else None,
        output_tokens=int(sum_optional("output_tokens")) if sum_optional("output_tokens") is not None else None,
        total_tokens=int(sum_optional("total_tokens")) if sum_optional("total_tokens") is not None else None,
        estimated_cost_usd=sum_optional("estimated_cost_usd"),
        latency_sec=sum_optional("latency_sec"),
    )


def placement_plan_quality_score(plan: Dict[str, Any], semantic: Dict[str, Any]) -> float:
    functional = normalize_functional_context(semantic.get("functional_context"))
    priority = functional.get("context_priority", 0.0)
    anchor = normalize_functional_anchor(plan.get("functional_anchor"))
    anchor_conf = anchor.get("confidence", 0.0)
    context_score = clamp01(plan.get("context_score", 0.0))
    placement_conf = clamp01(plan.get("placement_confidence", 0.0))

    # When functional context is important, context/anchor quality dominates.
    contextual = 0.55 * context_score + 0.45 * anchor_conf
    return float((1.0 - 0.55 * priority) * placement_conf + (0.55 * priority) * contextual)


def placement_context_is_acceptable(
    plan: Dict[str, Any],
    semantic: Dict[str, Any],
    anchor_confidence_min: float,
    context_score_min: float,
) -> bool:
    functional = normalize_functional_context(semantic.get("functional_context"))
    if (
        functional.get("context_priority", 0.0) < 0.35
        or str(functional.get("primary_anchor_type", "NONE")).upper() == "NONE"
    ):
        return True

    anchor = normalize_functional_anchor(plan.get("functional_anchor"))
    return bool(
        str(anchor.get("anchor_type", "NONE")).upper() != "NONE"
        and float(anchor.get("confidence", 0.0)) >= anchor_confidence_min
        and float(plan.get("context_score", 0.0)) >= context_score_min
    )

def normalize_placement_plan(
    raw: Dict[str, Any],
    semantic_strategy: str,
) -> Dict[str, Any]:

    allowed = {
        "PLANAR_SURFACE", "GROUNDED_OBJECT", "SURFACE_OBJECT",
        "WALL_MOUNTED", "HANGING_OBJECT", "GENERIC"
    }

    strategy = str(raw.get("placement_strategy", semantic_strategy)).upper()
    if strategy not in allowed:
        strategy = semantic_strategy if semantic_strategy in allowed else "GENERIC"

    try:
        confidence = clamp01(float(raw.get("placement_confidence", 0.0)))
    except Exception:
        confidence = 0.0

    try:
        context_score = clamp01(float(raw.get("context_score", 0.0)))
    except Exception:
        context_score = 0.0

    quad = normalize_quad(raw.get("product_quad"))
    bbox = normalize_bbox(raw.get("product_bbox"))
    contact = raw.get("contact_line", [])

    if isinstance(contact, list) and len(contact) == 2:
        contact = [normalize_point(contact[0]), normalize_point(contact[1])]
    else:
        contact = []

    valid = len(quad) == 4 if strategy in {"PLANAR_SURFACE", "WALL_MOUNTED"} else len(bbox) == 4
    if not valid:
        fallback = fallback_plan_for_strategy(strategy)
        fallback["functional_anchor"] = normalize_functional_anchor(raw.get("functional_anchor"))
        fallback["context_score"] = context_score
        return fallback

    return {
        "placement_strategy": strategy,
        "support_surface_type": str(raw.get("support_surface_type", "")).strip(),
        "placement_confidence": confidence,
        "context_score": context_score,
        "functional_anchor": normalize_functional_anchor(raw.get("functional_anchor")),
        "product_quad": quad,
        "product_bbox": bbox,
        "contact_line": contact,
        "surface_perspective": normalize_surface_perspective(
            raw.get("surface_perspective"),
            strategy=strategy,
        ),
        "rationale": str(raw.get("rationale", "")).strip(),
    }

def build_image_generation_config(
    aspect_ratio: str,
    temperature: float,
    jpeg_quality: int,
) -> types.GenerateContentConfig:

    return types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=temperature,
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size="4K",
            output_mime_type="image/jpeg",
            output_compression_quality=jpeg_quality,
        ),
    )


def generate_background_with_retry(
    client: Any,
    mode: str,
    model_id: str,
    prompt: str,
    config: types.GenerateContentConfig,
) -> Dict[str, Any]:

    start_all = time.perf_counter()
    retry_wait = 0.0
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start_attempt = time.perf_counter()

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )

            latency = time.perf_counter() - start_attempt
            e2e = time.perf_counter() - start_all

            image_bytes, mime = extract_image_bytes(response)

            if not image_bytes:
                raise RuntimeError("Image model returned no image.")

            raw_tt, normalized_tt = extract_traffic_type(response)
            usage = extract_usage(response)
            output_image_tokens = extract_image_output_tokens(response)

            return {
                "ok": True,
                "response": response,
                "image_bytes": image_bytes,
                "image_mime": mime,
                "traffic_type_raw": raw_tt,
                "traffic_type_normalized": normalized_tt,
                "usage": usage,
                "output_image_tokens": output_image_tokens,
                "attempts_used": attempt,
                "retry_count": attempt - 1,
                "retry_wait_sec": retry_wait,
                "latency_sec": latency,
                "end_to_end_latency_sec": e2e,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            code = http_code_from_exception(exc)

            if attempt >= MAX_ATTEMPTS or not is_retryable_api_exception(exc):
                break

            waits = RETRY_BACKOFF[mode]
            wait = waits[min(attempt - 1, len(waits) - 1)]

            label = f"HTTP {code}" if code is not None else type(exc).__name__
            print(f"  Retryable {label}; wait {wait}s...")
            time.sleep(wait)
            retry_wait += wait

    return {
        "ok": False,
        "response": None,
        "image_bytes": None,
        "image_mime": "",
        "traffic_type_raw": "",
        "traffic_type_normalized": "",
        "usage": {},
        "output_image_tokens": None,
        "attempts_used": MAX_ATTEMPTS,
        "retry_count": MAX_ATTEMPTS - 1,
        "retry_wait_sec": retry_wait,
        "latency_sec": None,
        "end_to_end_latency_sec": time.perf_counter() - start_all,
        "error": last_error,
    }



def build_final_integration_prompt(
    background_description: str,
    semantic: Dict[str, Any],
    plan: Dict[str, Any],
) -> str:
    semantic_json = json.dumps(semantic, ensure_ascii=False, indent=2)
    plan_json = json.dumps(
        {
            "placement_strategy": plan.get("placement_strategy"),
            "support_surface_type": plan.get("support_surface_type"),
            "functional_anchor": plan.get("functional_anchor"),
            "context_score": plan.get("context_score"),
            "product_quad": plan.get("product_quad"),
            "product_bbox": plan.get("product_bbox"),
            "surface_perspective": plan.get("surface_perspective"),
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are doing a final photoreal integration pass for an ecommerce product background replacement.

Inputs are provided in this order:
1. ORIGINAL_PRODUCT_REFERENCE: exact product identity/artwork/colors/text/proportions/material.
2. CLEAN_BACKGROUND: generated scene without the product.
3. DETERMINISTIC_COMPOSITE: APPROVED layout, scale, perspective, anchor-relative position, and product pixels.
4. PRODUCT_ALPHA_MASK: WHITE is product interior; BLACK is non-product.

Scene request:
{background_description}

Product semantics:
{semantic_json}

Approved placement plan:
{plan_json}

PRODUCT IDENTITY AND LAYOUT LOCK:
- Preserve product identity, artwork layout, readable text, silhouette, proportions, and material character.
- Do NOT replace, reword, simplify, beautify, redesign, or invent product artwork/text/logo/details.
- Do NOT move the product away from the approved anchor-relative placement.
- Keep product scale, orientation, crop, and overall aspect ratio consistent with DETERMINISTIC_COMPOSITE.
- Allow scene-consistent relighting, white balance, material blending, texture/noise matching, edge integration, and perspective-consistent contact if these improve photorealism without changing the intended product design.

Allowed improvements:
- physically correct contact shadow immediately outside/under the product,
- ambient occlusion at support contact,
- subtle reflected light / color spill near the silhouette,
- very subtle edge blending,
- scene-consistent noise/compression,
- background-side reflections or floor interaction,
- low-frequency exposure/white-balance harmonization that keeps product identity and artwork legible.

Functional placement lock:
- Preserve the approved relationship to the detected functional anchor.
- Do not move the product toward a visually emptier region.
- Do not add, remove, move, or obscure the functional anchor.

Background lock:
- Preserve CLEAN_BACKGROUND content and camera framing.
- Do not add duplicate products, rugs/mats, furniture, props, people, animals, decorative text, or seasonal objects.
- For planar products, keep them flush with the support plane with contact darkening, not a floating drop shadow.

Output only the final photoreal image.
""".strip()

def alpha_mask_preview(placed_product: Image.Image) -> Image.Image:
    alpha = placed_product.convert("RGBA").getchannel("A")
    return Image.merge("RGB", (alpha, alpha, alpha))


def masked_rgb_mae(
    reference: Image.Image,
    candidate: Image.Image,
    alpha: Image.Image,
) -> Optional[float]:
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.LANCZOS)
    if alpha.size != reference.size:
        alpha = alpha.resize(reference.size, Image.LANCZOS)

    a = np.asarray(reference.convert("RGB"), dtype=np.float32)
    b = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    m = np.asarray(alpha.convert("L"), dtype=np.uint8) > 64

    if not np.any(m):
        return None

    return float(np.abs(a[m] - b[m]).mean())


def generate_final_integration_with_retry(
    client: Any,
    mode: str,
    model_id: str,
    prompt: str,
    product_reference: Image.Image,
    clean_background: Image.Image,
    deterministic_composite: Image.Image,
    product_alpha_mask: Image.Image,
    config: types.GenerateContentConfig,
) -> Dict[str, Any]:

    start_all = time.perf_counter()
    retry_wait = 0.0
    last_error = None

    content = types.Content(
        role="user",
        parts=[
            image_part(product_reference, max_bytes=3_500_000),
            image_part(clean_background, max_bytes=3_500_000),
            image_part(deterministic_composite, max_bytes=4_500_000),
            image_part(product_alpha_mask, max_bytes=1_000_000),
            types.Part.from_text(text=prompt),
        ],
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start_attempt = time.perf_counter()

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[content],
                config=config,
            )

            latency = time.perf_counter() - start_attempt
            e2e = time.perf_counter() - start_all

            image_bytes, mime = extract_image_bytes(response)

            if not image_bytes:
                raise RuntimeError("Final integration model returned no image.")

            raw_tt, normalized_tt = extract_traffic_type(response)
            usage = extract_usage(response)
            output_image_tokens = extract_image_output_tokens(response)

            return {
                "ok": True,
                "response": response,
                "image_bytes": image_bytes,
                "image_mime": mime,
                "traffic_type_raw": raw_tt,
                "traffic_type_normalized": normalized_tt,
                "usage": usage,
                "output_image_tokens": output_image_tokens,
                "attempts_used": attempt,
                "retry_count": attempt - 1,
                "retry_wait_sec": retry_wait,
                "latency_sec": latency,
                "end_to_end_latency_sec": e2e,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            code = http_code_from_exception(exc)

            if attempt >= MAX_ATTEMPTS or not is_retryable_api_exception(exc):
                break

            waits = RETRY_BACKOFF[mode]
            wait = waits[min(attempt - 1, len(waits) - 1)]

            label = f"HTTP {code}" if code is not None else type(exc).__name__
            print(f"  Retryable final-integration {label}; wait {wait}s...")
            time.sleep(wait)
            retry_wait += wait

    return {
        "ok": False,
        "response": None,
        "image_bytes": None,
        "image_mime": "",
        "traffic_type_raw": "",
        "traffic_type_normalized": "",
        "usage": {},
        "output_image_tokens": None,
        "attempts_used": MAX_ATTEMPTS,
        "retry_count": MAX_ATTEMPTS - 1,
        "retry_wait_sec": retry_wait,
        "latency_sec": None,
        "end_to_end_latency_sec": time.perf_counter() - start_all,
        "error": last_error,
    }


def estimate_image_cost(
    mode: str,
    model_key: str,
    prompt_tokens: Optional[int],
    thinking_tokens: Optional[int],
    output_image_tokens: Optional[int],
    pricing_verified: bool,
) -> Optional[float]:

    if not pricing_verified:
        return None

    if prompt_tokens is None or output_image_tokens is None:
        return None

    p = IMAGE_PRICING[mode][model_key]

    return (
        prompt_tokens / 1_000_000 * p["input"]
        + (thinking_tokens or 0) / 1_000_000 * p["text_output"]
        + output_image_tokens / 1_000_000 * p["image_output"]
    )


def preview_output_only_cost(modes: List[str], models: List[str]) -> float:
    total = 0.0

    for mode in modes:
        for key in models:
            total += (
                OUTPUT_IMAGE_TOKENS_4K[key]
                / 1_000_000
                * IMAGE_PRICING[mode][key]["image_output"]
            )

    return total


# ============================================================
# 14. DETERMINISTIC PLACEMENT ENGINE
# ============================================================

def normalized_quad_to_pixels(
    quad: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:

    return np.array(
        [[p[0] * width, p[1] * height] for p in quad],
        dtype=np.float32
    )



def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def quad_metrics(quad_px: np.ndarray) -> Dict[str, float]:
    q = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    tl, tr, br, bl = q

    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))

    avg_w = max(1e-6, (top_w + bottom_w) * 0.5)
    avg_h = max(1e-6, (left_h + right_h) * 0.5)

    return {
        "top_width": top_w,
        "bottom_width": bottom_w,
        "left_height": left_h,
        "right_height": right_h,
        "avg_width": avg_w,
        "avg_height": avg_h,
        "width_height_ratio": avg_w / avg_h,
        "height_width_ratio": avg_h / avg_w,
        "area_px": polygon_area(q),
        "top_bottom_width_ratio": top_w / max(bottom_w, 1e-6),
        "left_right_height_ratio": left_h / max(right_h, 1e-6),
    }


def source_product_geometry(
    crop_rgba: Image.Image,
    source_quad: np.ndarray,
    semantic: Dict[str, Any],
) -> Dict[str, Any]:
    crop_w, crop_h = crop_rgba.size
    metrics = quad_metrics(source_quad)

    return {
        "crop_width_px": int(crop_w),
        "crop_height_px": int(crop_h),
        "crop_aspect_ratio_w_over_h": float(crop_w / max(crop_h, 1)),
        "visible_quad_aspect_ratio_w_over_h": float(metrics["width_height_ratio"]),
        "natural_orientation": semantic.get("natural_orientation", "UNKNOWN"),
        "perspective_transform_allowed": bool(semantic.get("perspective_transform_allowed", False)),
    }


def _quad_axis_unit_vectors(quad_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    tl, tr, br, bl = q

    width_vec = (tr - tl) + (br - bl)
    height_vec = (bl - tl) + (br - tr)

    def unit(v: np.ndarray, fallback: Tuple[float, float]) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array(fallback, dtype=np.float32)
        return (v / n).astype(np.float32)

    u = unit(width_vec, (1.0, 0.0))
    v = unit(height_vec, (0.0, 1.0))
    return u, v


def _anisotropic_scale_quad(
    quad_px: np.ndarray,
    scale_width: float = 1.0,
    scale_height: float = 1.0,
) -> np.ndarray:
    q = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    c = q.mean(axis=0)
    u, v = _quad_axis_unit_vectors(q)

    # Orthogonalize v against u so width/height scaling is stable even when
    # the AI returns a mildly skewed trapezoid.
    v = v - u * float(np.dot(v, u))
    vn = float(np.linalg.norm(v))
    if vn < 1e-6:
        v = np.array([-u[1], u[0]], dtype=np.float32)
    else:
        v = (v / vn).astype(np.float32)

    out = []
    for p in q:
        d = p - c
        du = float(np.dot(d, u))
        dv = float(np.dot(d, v))
        residual = d - du * u - dv * v
        out.append(c + (du * scale_width) * u + (dv * scale_height) * v + residual)

    return order_quad_points(np.asarray(out, dtype=np.float32))


def _fit_quad_inside_canvas(
    original_quad: np.ndarray,
    candidate_quad: np.ndarray,
    width: int,
    height: int,
    margin_norm: float,
) -> np.ndarray:
    """
    Keep the candidate inside a safe canvas margin without hard clipping corners.
    We binary-search between the original and candidate geometry.
    """
    margin_x = max(0.0, min(0.20, margin_norm)) * width
    margin_y = max(0.0, min(0.20, margin_norm)) * height

    def valid(q: np.ndarray) -> bool:
        return bool(
            np.all(q[:, 0] >= margin_x)
            and np.all(q[:, 0] <= width - margin_x)
            and np.all(q[:, 1] >= margin_y)
            and np.all(q[:, 1] <= height - margin_y)
            and polygon_area(q) > 16.0
        )

    if valid(candidate_quad):
        return order_quad_points(candidate_quad)

    lo, hi = 0.0, 1.0
    best = order_quad_points(original_quad)

    for _ in range(28):
        t = (lo + hi) * 0.5
        q = original_quad + (candidate_quad - original_quad) * t
        q = order_quad_points(q)
        if valid(q):
            best = q
            lo = t
        else:
            hi = t

    return best


def _uniform_scale_quad_to_area(
    quad_px: np.ndarray,
    target_area_ratio: float,
    canvas_w: int,
    canvas_h: int,
    margin_norm: float,
) -> np.ndarray:
    q = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    canvas_area = float(max(1, canvas_w * canvas_h))
    area = max(1.0, polygon_area(q))
    target_area = max(1.0, target_area_ratio * canvas_area)

    if area >= target_area:
        return q

    scale = math.sqrt(target_area / area)
    c = q.mean(axis=0)
    candidate = c + (q - c) * scale
    return _fit_quad_inside_canvas(q, candidate, canvas_w, canvas_h, margin_norm)


def _uniform_scale_quad_down_to_area(
    quad_px: np.ndarray,
    max_area_ratio: float,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    q = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    canvas_area = float(max(1, canvas_w * canvas_h))
    area = max(1.0, polygon_area(q))
    max_area = max(1.0, max_area_ratio * canvas_area)

    if area <= max_area:
        return q

    scale = math.sqrt(max_area / area)
    c = q.mean(axis=0)
    return order_quad_points(c + (q - c) * scale)


def _unit_vec(v: np.ndarray, fallback: Tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (v / n).astype(np.float32)


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit_vec(a)
    ub = _unit_vec(b)
    dot = max(-1.0, min(1.0, float(np.dot(ua, ub))))
    return float(math.degrees(math.acos(abs(dot))))


def _is_convex_quad(q: np.ndarray) -> bool:
    q = order_quad_points(np.asarray(q, dtype=np.float32))
    crosses = []
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i + 2) % 4] - q[(i + 1) % 4]
        crosses.append(float(np.cross(a, b)))
    pos = any(c > 1e-4 for c in crosses)
    neg = any(c < -1e-4 for c in crosses)
    return not (pos and neg)


def _regularize_floor_perspective(
    quad_px: np.ndarray,
    canvas_size: Tuple[int, int],
    surface_perspective: Dict[str, Any],
    guard_strength: float = 0.85,
    max_near_far_ratio: float = 1.35,
    max_opposite_edge_angle_delta_deg: float = 12.0,
    canvas_margin: float = 0.025,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generic floor-plane perspective regularizer.

    It does not detect product categories. It only runs when the placement strategy is
    planar and the analyzed support plane is a floor (or the semantic natural support
    is a floor in the caller).

    The regularizer preserves the AI-selected center and near edge as much as possible,
    while nudging the far/near edge scale relationship toward the perspective measured
    from the generated background.
    """
    canvas_w, canvas_h = canvas_size
    q0 = order_quad_points(np.asarray(quad_px, dtype=np.float32))
    q = q0.copy()

    sp = surface_perspective or {}
    farther = str(sp.get("farther_image_direction", "NONE")).upper()
    confidence = clamp01(float(sp.get("confidence", 0.0) or 0.0))
    persp_strength = clamp01(float(sp.get("perspective_strength", 0.0) or 0.0))
    hint_ratio = float(sp.get("near_far_scale_ratio", 1.0) or 1.0)

    guard_strength = clamp01(guard_strength)
    max_near_far_ratio = max(1.0, min(float(max_near_far_ratio), 1.60))
    max_angle = max(1.0, min(float(max_opposite_edge_angle_delta_deg), 45.0))
    hint_ratio = max(1.0, min(hint_ratio, max_near_far_ratio))

    tl, tr, br, bl = q
    top_vec = tr - tl
    bottom_vec = br - bl
    top_w = float(np.linalg.norm(top_vec))
    bottom_w = float(np.linalg.norm(bottom_vec))

    changes: List[str] = []
    enabled = farther in {"UP", "DOWN"} and confidence >= 0.20

    before_near_far = 1.0
    if farther == "UP":
        before_near_far = bottom_w / max(top_w, 1e-6)
    elif farther == "DOWN":
        before_near_far = top_w / max(bottom_w, 1e-6)

    if not enabled:
        return q, {
            "enabled": False,
            "reason": "insufficient or unsupported surface-perspective signal",
            "farther_image_direction": farther,
            "surface_confidence": confidence,
            "before_near_far_ratio": before_near_far,
            "after_near_far_ratio": before_near_far,
            "changes": changes,
        }

    # Blend the model hint according to both plane confidence and perspective strength.
    # A near-overhead scene therefore remains almost rectangular.
    signal = guard_strength * confidence * max(0.15, persp_strength)
    desired_ratio = 1.0 + (hint_ratio - 1.0) * signal

    # If the AI quad tapers in the physically opposite direction, correct more strongly.
    if before_near_far < 1.0:
        desired_ratio = max(desired_ratio, 1.0 + 0.04 * guard_strength * confidence)

    desired_ratio = max(1.0, min(desired_ratio, max_near_far_ratio))

    # Opposite edges should represent the same planar width direction locally.
    edge_angle_delta = _angle_between_deg(top_vec, bottom_vec)
    u_top = _unit_vec(top_vec)
    u_bottom = _unit_vec(bottom_vec)
    if float(np.dot(u_top, u_bottom)) < 0:
        u_bottom = -u_bottom
    u_avg = _unit_vec(u_top + u_bottom, fallback=(1.0, 0.0))

    tc = (tl + tr) * 0.5
    bc = (bl + br) * 0.5

    if farther == "UP":
        near_w = bottom_w
        target_far_w = near_w / max(desired_ratio, 1e-6)
        blend = clamp01(guard_strength * confidence)
        new_top_w = top_w + (target_far_w - top_w) * blend
        new_bottom_w = bottom_w
    else:  # farther == DOWN
        near_w = top_w
        target_far_w = near_w / max(desired_ratio, 1e-6)
        blend = clamp01(guard_strength * confidence)
        new_bottom_w = bottom_w + (target_far_w - bottom_w) * blend
        new_top_w = top_w

    # If the two width edges are twisted relative to each other, align their local
    # directions toward the average axis. This fixes many "sticker on floor" quads
    # without forcing a global horizontal edge.
    dir_blend = 0.0
    if edge_angle_delta > max_angle:
        dir_blend = clamp01((edge_angle_delta - max_angle) / max(edge_angle_delta, 1e-6))
        dir_blend *= clamp01(guard_strength * confidence)
        changes.append(f"aligned_opposite_width_edges:{edge_angle_delta:.2f}deg")

    def blend_dir(original: np.ndarray) -> np.ndarray:
        u = _unit_vec(original)
        if float(np.dot(u, u_avg)) < 0:
            u = -u
        return _unit_vec(u * (1.0 - dir_blend) + u_avg * dir_blend)

    top_dir = blend_dir(top_vec)
    bottom_dir = blend_dir(bottom_vec)

    candidate = np.stack([
        tc - top_dir * (new_top_w * 0.5),
        tc + top_dir * (new_top_w * 0.5),
        bc + bottom_dir * (new_bottom_w * 0.5),
        bc - bottom_dir * (new_bottom_w * 0.5),
    ]).astype(np.float32)
    candidate = order_quad_points(candidate)

    if _is_convex_quad(candidate) and polygon_area(candidate) > 16.0:
        fitted = _fit_quad_inside_canvas(q, candidate, canvas_w, canvas_h, canvas_margin)
        if not np.allclose(fitted, q):
            q = fitted
            changes.append(f"regularized_floor_taper:{before_near_far:.3f}->{desired_ratio:.3f}")

    qm = quad_metrics(q)
    if farther == "UP":
        after_near_far = qm["bottom_width"] / max(qm["top_width"], 1e-6)
    else:
        after_near_far = qm["top_width"] / max(qm["bottom_width"], 1e-6)

    return q, {
        "enabled": True,
        "farther_image_direction": farther,
        "surface_confidence": confidence,
        "perspective_strength": persp_strength,
        "near_far_scale_ratio_hint": hint_ratio,
        "guard_strength": guard_strength,
        "max_near_far_ratio": max_near_far_ratio,
        "max_opposite_edge_angle_delta_deg": max_angle,
        "before_near_far_ratio": before_near_far,
        "after_near_far_ratio": after_near_far,
        "opposite_edge_angle_delta_deg": edge_angle_delta,
        "changes": changes,
    }


def _ensure_dominant_orientation(
    q: np.ndarray,
    orientation: str,
    src_ratio: float,
    canvas_w: int,
    canvas_h: int,
    orientation_preservation: float,
    minimum_dominant_ratio: float,
    canvas_margin: float,
) -> Tuple[np.ndarray, List[str]]:
    """Apply the source-orientation guard without touching perspective taper."""
    q = order_quad_points(np.asarray(q, dtype=np.float32))
    metrics = quad_metrics(q)
    changes: List[str] = []

    if orientation == "HORIZONTAL":
        desired_min = max(minimum_dominant_ratio, src_ratio * orientation_preservation)
        current = metrics["width_height_ratio"]
        if current < desired_min:
            scale_w = desired_min / max(current, 1e-6)
            candidate = _anisotropic_scale_quad(q, scale_width=scale_w, scale_height=1.0)
            q2 = _fit_quad_inside_canvas(q, candidate, canvas_w, canvas_h, canvas_margin)
            if not np.allclose(q2, q):
                q = q2
                changes.append(f"expanded_width_for_horizontal_orientation:{scale_w:.3f}x")

    elif orientation == "VERTICAL":
        source_h_over_w = 1.0 / max(src_ratio, 1e-6)
        desired_min = max(minimum_dominant_ratio, source_h_over_w * orientation_preservation)
        current = metrics["height_width_ratio"]
        if current < desired_min:
            scale_h = desired_min / max(current, 1e-6)
            candidate = _anisotropic_scale_quad(q, scale_width=1.0, scale_height=scale_h)
            q2 = _fit_quad_inside_canvas(q, candidate, canvas_w, canvas_h, canvas_margin)
            if not np.allclose(q2, q):
                q = q2
                changes.append(f"expanded_height_for_vertical_orientation:{scale_h:.3f}x")

    return q, changes


def stabilize_planar_quad(
    quad_norm: List[List[float]],
    canvas_size: Tuple[int, int],
    semantic: Dict[str, Any],
    product_geometry: Dict[str, Any],
    surface_perspective: Optional[Dict[str, Any]] = None,
    min_planar_area_ratio: float = 0.10,
    max_planar_area_ratio: float = 0.32,
    orientation_preservation: float = 0.90,
    minimum_dominant_ratio: float = 1.20,
    canvas_margin: float = 0.025,
    perspective_guard_strength: float = 0.85,
    max_near_far_ratio: float = 1.35,
    max_opposite_edge_angle_delta_deg: float = 12.0,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Best-practice generic geometry guard for planar product placement.

    Order of operations matters:
      1) preserve source dominant orientation,
      2) regularize floor-plane perspective using the generated background analysis,
      3) re-check orientation after perspective correction,
      4) clamp ecommerce prominence by projected area,
      5) keep the quad safely inside the canvas.
    """
    canvas_w, canvas_h = canvas_size
    q0 = order_quad_points(normalized_quad_to_pixels(quad_norm, canvas_w, canvas_h))
    q = q0.copy()

    before = quad_metrics(q)
    orientation = str(semantic.get("natural_orientation", "UNKNOWN")).upper()
    natural_support = str(semantic.get("natural_support", "UNKNOWN")).upper()

    src_ratio = float(product_geometry.get("crop_aspect_ratio_w_over_h", 1.0) or 1.0)
    src_ratio = max(0.05, min(src_ratio, 20.0))

    orientation_preservation = max(0.50, min(float(orientation_preservation), 1.0))
    minimum_dominant_ratio = max(1.01, float(minimum_dominant_ratio))
    min_planar_area_ratio = max(0.0, min(float(min_planar_area_ratio), 0.80))
    max_planar_area_ratio = max(min_planar_area_ratio, min(float(max_planar_area_ratio), 0.90))

    changes: List[str] = []

    q, c = _ensure_dominant_orientation(
        q=q,
        orientation=orientation,
        src_ratio=src_ratio,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        orientation_preservation=orientation_preservation,
        minimum_dominant_ratio=minimum_dominant_ratio,
        canvas_margin=canvas_margin,
    )
    changes.extend(c)

    perspective_guard: Dict[str, Any] = {
        "enabled": False,
        "reason": "not a floor planar placement",
        "changes": [],
    }

    # The support-specific rule is generic: any planar product whose natural support
    # is FLOOR/GROUND can use this floor-plane guard. No product-category check exists.
    if natural_support in {"FLOOR", "GROUND"}:
        q, perspective_guard = _regularize_floor_perspective(
            quad_px=q,
            canvas_size=canvas_size,
            surface_perspective=surface_perspective or {},
            guard_strength=perspective_guard_strength,
            max_near_far_ratio=max_near_far_ratio,
            max_opposite_edge_angle_delta_deg=max_opposite_edge_angle_delta_deg,
            canvas_margin=canvas_margin,
        )
        changes.extend(perspective_guard.get("changes", []))

        # Perspective correction can slightly alter visible W/H, so re-assert the
        # product's dominant orientation using a uniform width/height-axis scale.
        q, c = _ensure_dominant_orientation(
            q=q,
            orientation=orientation,
            src_ratio=src_ratio,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            orientation_preservation=orientation_preservation,
            minimum_dominant_ratio=minimum_dominant_ratio,
            canvas_margin=canvas_margin,
        )
        changes.extend(c)

    # Ecommerce prominence guard.
    area_ratio = polygon_area(q) / float(max(1, canvas_w * canvas_h))
    if area_ratio < min_planar_area_ratio:
        old = q.copy()
        q = _uniform_scale_quad_to_area(
            q,
            target_area_ratio=min_planar_area_ratio,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            margin_norm=canvas_margin,
        )
        if not np.allclose(old, q):
            changes.append(f"enlarged_to_min_area:{min_planar_area_ratio:.3f}")

    area_ratio = polygon_area(q) / float(max(1, canvas_w * canvas_h))
    if area_ratio > max_planar_area_ratio:
        q = _uniform_scale_quad_down_to_area(
            q,
            max_area_ratio=max_planar_area_ratio,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        changes.append(f"reduced_to_max_area:{max_planar_area_ratio:.3f}")

    q = order_quad_points(q)
    after = quad_metrics(q)

    q_norm = [
        [clamp01(float(x) / canvas_w), clamp01(float(y) / canvas_h)]
        for x, y in q
    ]

    guard = {
        "enabled": True,
        "natural_orientation": orientation,
        "natural_support": natural_support,
        "source_crop_aspect_ratio_w_over_h": src_ratio,
        "before_width_height_ratio": before["width_height_ratio"],
        "after_width_height_ratio": after["width_height_ratio"],
        "before_area_ratio": before["area_px"] / float(max(1, canvas_w * canvas_h)),
        "after_area_ratio": after["area_px"] / float(max(1, canvas_w * canvas_h)),
        "orientation_preservation": orientation_preservation,
        "minimum_dominant_ratio": minimum_dominant_ratio,
        "min_planar_area_ratio": min_planar_area_ratio,
        "max_planar_area_ratio": max_planar_area_ratio,
        "canvas_margin": canvas_margin,
        "perspective_guard": perspective_guard,
        "changes": changes,
    }

    return q_norm, guard



def _angle_delta_mod_pi(target: float, current: float) -> float:
    """Smallest signed angle delta when line orientation is modulo 180 degrees."""
    return float((target - current + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _rotate_points_px(points: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    if abs(angle_rad) < 1e-8:
        return np.asarray(points, dtype=np.float32).copy()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    return ((pts - center[None, :]) @ R.T + center[None, :]).astype(np.float32)


def _anchor_reference_px(
    anchor: Dict[str, Any],
    canvas_size: Tuple[int, int],
    side: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (reference_point_px, anchor_line_px).
    The reference point is chosen from the placement-relevant anchor edge when possible.
    """
    w, h = canvas_size

    line = anchor.get("anchor_line", [])
    line_px = None
    if isinstance(line, list) and len(line) == 2:
        line_px = np.array(
            [[line[0][0] * w, line[0][1] * h],
             [line[1][0] * w, line[1][1] * h]],
            dtype=np.float32,
        )
        if float(np.linalg.norm(line_px[1] - line_px[0])) < 2.0:
            line_px = None

    bbox = anchor.get("anchor_bbox", [])
    ref = None
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if side == "DOWN":
            ref = np.array([(x1 + x2) * 0.5 * w, y2 * h], dtype=np.float32)
        elif side == "UP":
            ref = np.array([(x1 + x2) * 0.5 * w, y1 * h], dtype=np.float32)
        elif side == "LEFT":
            ref = np.array([x1 * w, (y1 + y2) * 0.5 * h], dtype=np.float32)
        elif side == "RIGHT":
            ref = np.array([x2 * w, (y1 + y2) * 0.5 * h], dtype=np.float32)
        else:
            ref = np.array([(x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h], dtype=np.float32)

    if ref is None and line_px is not None:
        ref = line_px.mean(axis=0)

    if ref is None:
        point = anchor.get("anchor_point", [])
        if isinstance(point, list) and len(point) == 2:
            ref = np.array([point[0] * w, point[1] * h], dtype=np.float32)

    return ref, line_px


def _infer_anchor_side(
    requested_side: str,
    anchor_ref_px: Optional[np.ndarray],
    product_center_px: np.ndarray,
) -> str:
    side = str(requested_side or "NONE").upper()
    if side in {"UP", "DOWN", "LEFT", "RIGHT", "OVERLAP"}:
        return side
    if anchor_ref_px is None:
        return "NONE"

    d = product_center_px - anchor_ref_px
    if abs(float(d[0])) > abs(float(d[1])):
        return "RIGHT" if d[0] >= 0 else "LEFT"
    return "DOWN" if d[1] >= 0 else "UP"



def _translate_quad_inside_canvas(
    candidate_quad: np.ndarray,
    width: int,
    height: int,
    margin_norm: float,
) -> np.ndarray:
    """Keep a translated/rotated quad inside the canvas without reducing desired Y/X movement together."""
    q = order_quad_points(np.asarray(candidate_quad, dtype=np.float32)).copy()
    margin = max(0.0, min(0.20, margin_norm))
    min_x = margin * width
    max_x = (1.0 - margin) * width
    min_y = margin * height
    max_y = (1.0 - margin) * height

    span_x = float(q[:, 0].max() - q[:, 0].min())
    span_y = float(q[:, 1].max() - q[:, 1].min())
    if span_x > (max_x - min_x) or span_y > (max_y - min_y):
        return q

    dx = 0.0
    dy = 0.0
    qminx, qmaxx = float(q[:, 0].min()), float(q[:, 0].max())
    qminy, qmaxy = float(q[:, 1].min()), float(q[:, 1].max())

    if qminx < min_x:
        dx = min_x - qminx
    elif qmaxx > max_x:
        dx = max_x - qmaxx

    if qminy < min_y:
        dy = min_y - qminy
    elif qmaxy > max_y:
        dy = max_y - qmaxy

    q[:, 0] += dx
    q[:, 1] += dy
    return order_quad_points(q)

def apply_functional_context_guard(
    plan: Dict[str, Any],
    semantic: Dict[str, Any],
    canvas_size: Tuple[int, int],
    strength: float = 0.90,
    anchor_confidence_min: float = 0.50,
    max_anchor_shift_ratio: float = 0.25,
    max_anchor_rotation_deg: float = 18.0,
    anchor_gap_ratio: float = 0.015,
    canvas_margin: float = 0.025,
) -> Dict[str, Any]:
    """
    Generic anchor-aware placement regularizer.

    It does not know product categories. It uses functional_context from semantic
    analysis plus a detected functional_anchor from the background analysis.
    """
    out = dict(plan)
    functional = normalize_functional_context(semantic.get("functional_context"))
    anchor = normalize_functional_anchor(out.get("functional_anchor"))

    priority = clamp01(functional.get("context_priority", 0.0))
    context_score = clamp01(out.get("context_score", 0.0))
    anchor_conf = clamp01(anchor.get("confidence", 0.0))
    strength = clamp01(strength)

    guard = {
        "enabled": False,
        "anchor_type": anchor.get("anchor_type", "NONE"),
        "semantic_anchor_type": functional.get("primary_anchor_type", "NONE"),
        "anchor_confidence": anchor_conf,
        "context_priority": priority,
        "context_score": context_score,
        "changes": [],
    }

    if (
        strength <= 0.0
        or priority < 0.25
        or anchor_conf < anchor_confidence_min
        or str(anchor.get("anchor_type", "NONE")).upper() == "NONE"
    ):
        guard["reason"] = "functional anchor unavailable or low-confidence"
        out["functional_guard"] = guard
        out["functional_anchor"] = anchor
        return out

    w, h = canvas_size
    strategy = out.get("placement_strategy", "GENERIC")
    effective = strength * anchor_conf * max(priority, context_score, 0.35)
    effective = clamp01(effective)
    # High-confidence functional relationships should dominate arbitrary empty-space
    # composition. This is what prevents an entrance product from drifting deep
    # into the room when the correct anchor is clearly visible.
    if priority >= 0.80 and anchor_conf >= 0.80 and context_score >= 0.65:
        effective = max(effective, 0.92)

    semantic_relation = functional.get("anchor_relation", "NONE")
    relation = anchor.get("relation", "NONE")
    if relation == "NONE":
        relation = semantic_relation

    alignment = anchor.get("preferred_alignment", "NONE")
    if alignment == "NONE":
        alignment = functional.get("preferred_alignment", "NONE")

    distance = functional.get("preferred_distance", "FLEXIBLE")
    distance_multiplier = {
        "TOUCHING": 0.25,
        "VERY_CLOSE": 0.65,
        "CLOSE": 1.0,
        "MODERATE": 2.0,
        "FLEXIBLE": 1.25,
    }.get(distance, 1.25)
    gap_px = max(2.0, min(w, h) * max(0.0, anchor_gap_ratio) * distance_multiplier)

    if strategy in {"PLANAR_SURFACE", "WALL_MOUNTED"} and len(out.get("product_quad", [])) == 4:
        q0 = normalized_quad_to_pixels(out["product_quad"], w, h)
        q0 = order_quad_points(q0)
        q = q0.copy()
        metrics = quad_metrics(q)
        center0 = q.mean(axis=0)

        side = _infer_anchor_side(
            anchor.get("placement_side", "NONE"),
            _anchor_reference_px(anchor, canvas_size, "NONE")[0],
            center0,
        )
        anchor_ref, anchor_line = _anchor_reference_px(anchor, canvas_size, side)

        if anchor_ref is None:
            guard["reason"] = "anchor geometry missing"
            out["functional_guard"] = guard
            out["functional_anchor"] = anchor
            return out

        guard["placement_side"] = side
        guard["before_center_norm"] = [float(center0[0] / w), float(center0[1] / h)]

        # Alignment correction. Use line orientation only when the anchor provides
        # a meaningful edge/axis. Rotation is clamped to avoid fighting the floor
        # plane or introducing dramatic geometry changes.
        if anchor_line is not None and alignment in {"PARALLEL", "PERPENDICULAR", "EDGE_ALIGNED"}:
            line_vec = anchor_line[1] - anchor_line[0]
            target_angle = math.atan2(float(line_vec[1]), float(line_vec[0]))
            if alignment == "PERPENDICULAR":
                target_angle += math.pi / 2.0

            u, _ = _quad_axis_unit_vectors(q)
            current_angle = math.atan2(float(u[1]), float(u[0]))
            delta = _angle_delta_mod_pi(target_angle, current_angle)
            max_rot = math.radians(max(0.0, max_anchor_rotation_deg))
            delta = max(-max_rot, min(max_rot, delta))
            applied_delta = delta * effective

            if abs(applied_delta) > math.radians(0.25):
                q = _rotate_points_px(q, q.mean(axis=0), applied_delta)
                guard["changes"].append("anchor_alignment")
                guard["rotation_deg"] = float(math.degrees(applied_delta))

        q = order_quad_points(q)
        metrics = quad_metrics(q)
        center = q.mean(axis=0)
        half_w = metrics["avg_width"] * 0.5
        half_h = metrics["avg_height"] * 0.5

        desired = center.copy()
        if relation in {"IN_FRONT_OF", "BELOW", "ABOVE", "NEXT_TO", "NEAR"} or side in {"UP", "DOWN", "LEFT", "RIGHT"}:
            if side == "DOWN":
                desired = np.array([anchor_ref[0], anchor_ref[1] + half_h + gap_px], dtype=np.float32)
            elif side == "UP":
                desired = np.array([anchor_ref[0], anchor_ref[1] - half_h - gap_px], dtype=np.float32)
            elif side == "LEFT":
                desired = np.array([anchor_ref[0] - half_w - gap_px, anchor_ref[1]], dtype=np.float32)
            elif side == "RIGHT":
                desired = np.array([anchor_ref[0] + half_w + gap_px, anchor_ref[1]], dtype=np.float32)
        elif relation in {"CENTERED_ON", "ON", "INSIDE", "ATTACHED_TO"} or side == "OVERLAP":
            desired = anchor_ref.astype(np.float32)

        shift = (desired - center) * effective
        max_shift_px = max(4.0, min(w, h) * max(0.01, max_anchor_shift_ratio))
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm > max_shift_px:
            shift *= max_shift_px / max(shift_norm, 1e-6)

        if float(np.linalg.norm(shift)) > 1.0:
            q_candidate = q + shift[None, :]
            q = _translate_quad_inside_canvas(
                candidate_quad=q_candidate,
                width=w,
                height=h,
                margin_norm=canvas_margin,
            )
            guard["changes"].append("anchor_position")

        q = order_quad_points(q)
        out["product_quad_context_original"] = out.get("product_quad")
        out["product_quad"] = [
            [clamp01(float(p[0] / w)), clamp01(float(p[1] / h))]
            for p in q
        ]
        out["contact_line"] = [
            out["product_quad"][3],
            out["product_quad"][2],
        ]

        center1 = q.mean(axis=0)
        guard["after_center_norm"] = [float(center1[0] / w), float(center1[1] / h)]
        guard["shift_px"] = float(np.linalg.norm(center1 - center0))
        guard["enabled"] = True
        guard["reason"] = "anchor-aware planar placement regularization"

    elif len(out.get("product_bbox", [])) == 4:
        x1, y1, x2, y2 = out["product_bbox"]
        center0 = np.array([(x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h], dtype=np.float32)
        bw = (x2 - x1) * w
        bh = (y2 - y1) * h

        side = _infer_anchor_side(
            anchor.get("placement_side", "NONE"),
            _anchor_reference_px(anchor, canvas_size, "NONE")[0],
            center0,
        )
        anchor_ref, _ = _anchor_reference_px(anchor, canvas_size, side)
        if anchor_ref is not None:
            desired = center0.copy()
            if side == "DOWN":
                desired = np.array([anchor_ref[0], anchor_ref[1] + bh * 0.5 + gap_px], dtype=np.float32)
            elif side == "UP":
                desired = np.array([anchor_ref[0], anchor_ref[1] - bh * 0.5 - gap_px], dtype=np.float32)
            elif side == "LEFT":
                desired = np.array([anchor_ref[0] - bw * 0.5 - gap_px, anchor_ref[1]], dtype=np.float32)
            elif side == "RIGHT":
                desired = np.array([anchor_ref[0] + bw * 0.5 + gap_px, anchor_ref[1]], dtype=np.float32)
            elif relation in {"CENTERED_ON", "ON", "INSIDE", "ATTACHED_TO"}:
                desired = anchor_ref.astype(np.float32)

            shift = (desired - center0) * effective
            max_shift_px = max(4.0, min(w, h) * max(0.01, max_anchor_shift_ratio))
            n = float(np.linalg.norm(shift))
            if n > max_shift_px:
                shift *= max_shift_px / max(n, 1e-6)

            nx1 = x1 + float(shift[0] / w)
            nx2 = x2 + float(shift[0] / w)
            ny1 = y1 + float(shift[1] / h)
            ny2 = y2 + float(shift[1] / h)

            margin = max(0.0, min(0.20, canvas_margin))
            dx = 0.0
            dy = 0.0
            if nx1 < margin:
                dx = margin - nx1
            elif nx2 > 1.0 - margin:
                dx = (1.0 - margin) - nx2
            if ny1 < margin:
                dy = margin - ny1
            elif ny2 > 1.0 - margin:
                dy = (1.0 - margin) - ny2

            out["product_bbox_context_original"] = out.get("product_bbox")
            out["product_bbox"] = [nx1 + dx, ny1 + dy, nx2 + dx, ny2 + dy]
            guard["enabled"] = True
            guard["placement_side"] = side
            guard["shift_px"] = float(np.linalg.norm(
                np.array([(out["product_bbox"][0] + out["product_bbox"][2]) * 0.5 * w,
                          (out["product_bbox"][1] + out["product_bbox"][3]) * 0.5 * h]) - center0
            ))
            guard["changes"].append("anchor_position")
            guard["reason"] = "anchor-aware object placement regularization"
        else:
            guard["reason"] = "anchor geometry missing"
    else:
        guard["reason"] = "placement geometry unavailable"

    out["functional_anchor"] = anchor
    out["functional_guard"] = guard
    return out

def stabilize_placement_plan(
    plan: Dict[str, Any],
    semantic: Dict[str, Any],
    product_geometry: Dict[str, Any],
    canvas_size: Tuple[int, int],
    min_planar_area_ratio: float,
    max_planar_area_ratio: float,
    orientation_preservation: float,
    minimum_dominant_ratio: float,
    canvas_margin: float,
    perspective_guard_strength: float,
    max_near_far_ratio: float,
    max_opposite_edge_angle_delta_deg: float,
) -> Dict[str, Any]:
    out = dict(plan)
    strategy = out.get("placement_strategy", "GENERIC")

    if strategy in {"PLANAR_SURFACE", "WALL_MOUNTED"} and len(out.get("product_quad", [])) == 4:
        corrected_quad, guard = stabilize_planar_quad(
            quad_norm=out["product_quad"],
            canvas_size=canvas_size,
            semantic=semantic,
            product_geometry=product_geometry,
            surface_perspective=out.get("surface_perspective", {}),
            min_planar_area_ratio=min_planar_area_ratio,
            max_planar_area_ratio=max_planar_area_ratio,
            orientation_preservation=orientation_preservation,
            minimum_dominant_ratio=minimum_dominant_ratio,
            canvas_margin=canvas_margin,
            perspective_guard_strength=perspective_guard_strength,
            max_near_far_ratio=max_near_far_ratio,
            max_opposite_edge_angle_delta_deg=max_opposite_edge_angle_delta_deg,
        )
        out["product_quad_original"] = out["product_quad"]
        out["product_quad"] = corrected_quad
        out["geometry_guard"] = guard

        xs = [p[0] for p in corrected_quad]
        ys = [p[1] for p in corrected_quad]
        out["product_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        out["contact_line"] = [corrected_quad[3], corrected_quad[2]]
    else:
        out["geometry_guard"] = {
            "enabled": False,
            "reason": "non-planar placement strategy",
        }

    return out



def save_placement_debug_overlay(
    background: Image.Image,
    plan: Dict[str, Any],
    output_path: Path,
) -> None:
    """Visual audit: AI proposal, context/geometry guarded placement, and anchor."""
    img = background.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_w = max(3, int(round(max(w, h) * 0.002)))

    def px_points(q):
        return [(int(round(p[0] * w)), int(round(p[1] * h))) for p in q]

    if plan.get("placement_strategy") in {"PLANAR_SURFACE", "WALL_MOUNTED"}:
        original = plan.get("product_quad_context_original") or plan.get("product_quad_original")
        final = plan.get("product_quad")
        if isinstance(original, list) and len(original) == 4:
            p = px_points(original)
            draw.line(p + [p[0]], fill=(220, 60, 60), width=line_w)
        if isinstance(final, list) and len(final) == 4:
            p = px_points(final)
            draw.line(p + [p[0]], fill=(60, 220, 90), width=line_w)

    anchor = normalize_functional_anchor(plan.get("functional_anchor"))
    abox = anchor.get("anchor_bbox", [])
    if isinstance(abox, list) and len(abox) == 4:
        x1, y1, x2, y2 = abox
        draw.rectangle(
            [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
            outline=(70, 140, 255),
            width=line_w,
        )

    aline = anchor.get("anchor_line", [])
    if isinstance(aline, list) and len(aline) == 2:
        p = px_points(aline)
        draw.line(p, fill=(40, 180, 255), width=line_w * 2)

    ap = anchor.get("anchor_point", [])
    if isinstance(ap, list) and len(ap) == 2:
        x, y = int(ap[0]*w), int(ap[1]*h)
        r = max(5, line_w * 2)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(40, 180, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=92, optimize=True)

def warp_planar_rgba(
    crop_rgba: Image.Image,
    src_quad: np.ndarray,
    dst_quad_px: np.ndarray,
    canvas_size: Tuple[int, int],
) -> Image.Image:

    src = np.array(crop_rgba.convert("RGBA"))
    H = cv2.getPerspectiveTransform(
        order_quad_points(src_quad).astype(np.float32),
        order_quad_points(dst_quad_px).astype(np.float32),
    )

    canvas_w, canvas_h = canvas_size

    warped = cv2.warpPerspective(
        src,
        H,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return Image.fromarray(warped, mode="RGBA")


def place_object_bbox(
    crop_rgba: Image.Image,
    bbox_norm: List[float],
    canvas_size: Tuple[int, int],
) -> Image.Image:

    canvas_w, canvas_h = canvas_size

    x1 = int(round(bbox_norm[0] * canvas_w))
    y1 = int(round(bbox_norm[1] * canvas_h))
    x2 = int(round(bbox_norm[2] * canvas_w))
    y2 = int(round(bbox_norm[3] * canvas_h))

    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    src_w, src_h = crop_rgba.size
    scale = min(target_w / src_w, target_h / src_h)

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = crop_rgba.resize((new_w, new_h), Image.LANCZOS)

    # center horizontally, bottom-align to support surface
    px = x1 + (target_w - new_w) // 2
    py = y2 - new_h

    out = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    out.alpha_composite(resized, (px, py))
    return out


def _alpha_bbox(alpha: np.ndarray, threshold: int = 16) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _local_background_sample_mask(alpha: np.ndarray) -> np.ndarray:
    """
    Select nearby visible background pixels for local lighting/color estimates.

    The sample intentionally stays near the placed object so a product on a warm
    floor receives the floor's exposure/tint rather than the whole room average.
    """
    h, w = alpha.shape
    sample = np.zeros((h, w), dtype=bool)

    bbox = _alpha_bbox(alpha)
    if bbox is None:
        return alpha < 8

    x1, y1, x2, y2 = bbox
    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)
    pad = max(16, int(round(max(obj_w, obj_h) * 0.22)))

    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(w, x2 + pad)
    sy2 = min(h, y2 + pad)
    sample[sy1:sy2, sx1:sx2] = True

    kernel_px = max(9, min(81, int(round(max(obj_w, obj_h) * 0.035))))
    if kernel_px % 2 == 0:
        kernel_px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    blocked = cv2.dilate((alpha > 8).astype(np.uint8), kernel, iterations=1).astype(bool)

    sample &= ~blocked

    if int(sample.sum()) < 200:
        sample = np.zeros((h, w), dtype=bool)
        sample[sy1:sy2, sx1:sx2] = True
        sample &= alpha < 8

    if int(sample.sum()) < 200:
        sample = alpha < 8

    return sample



def harmonize_product_with_background(
    background: Image.Image,
    placed_product: Image.Image,
    realism_strength: float,
    edge_light_wrap: float,
) -> Image.Image:
    """
    Deterministic, product-safe harmonization.

    Goals:
      - inherit low-frequency scene illumination,
      - adapt white balance/chromaticity very slightly,
      - reduce excessive catalog-vs-lifestyle saturation mismatch,
      - blend only a narrow silhouette band with low-frequency environment color,
      - never redraw product artwork/text.
    """
    strength = clamp01(realism_strength)
    wrap_strength = clamp01(edge_light_wrap)
    if strength <= 0.0 and wrap_strength <= 0.0:
        return placed_product.convert("RGBA")

    product = placed_product.convert("RGBA")
    rgba = np.asarray(product, dtype=np.float32).copy()
    alpha = rgba[..., 3].astype(np.uint8)
    mask = alpha > 16
    if not np.any(mask):
        return product

    bg_rgb = np.asarray(background.convert("RGB"), dtype=np.float32)
    sample_mask = _local_background_sample_mask(alpha)
    if not np.any(sample_mask):
        return product

    sample_rgb = bg_rgb[sample_mask]
    if sample_rgb.shape[0] > 100:
        lo = np.percentile(sample_rgb, 8, axis=0)
        hi = np.percentile(sample_rgb, 92, axis=0)
        keep = np.all((sample_rgb >= lo[None, :]) & (sample_rgb <= hi[None, :]), axis=1)
        if int(keep.sum()) > 100:
            sample_rgb = sample_rgb[keep]

    sample_median_rgb = np.median(sample_rgb, axis=0).astype(np.float32)
    sample_luma = (
        sample_rgb[:, 0] * 0.299
        + sample_rgb[:, 1] * 0.587
        + sample_rgb[:, 2] * 0.114
    )
    reference_luma = max(1.0, float(np.median(sample_luma)))

    # 1) Local illumination field: keep spatial falloff from windows/room lighting.
    sigma = max(8.0, min(120.0, max(alpha.shape) * 0.018))
    bg_blur = cv2.GaussianBlur(bg_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_luma = (
        bg_blur[..., 0] * 0.299
        + bg_blur[..., 1] * 0.587
        + bg_blur[..., 2] * 0.114
    )
    light_factor = np.clip(bg_luma / reference_luma, 0.84, 1.16)
    light_factor = 1.0 + (light_factor - 1.0) * (0.82 * strength)
    rgba[..., :3][mask] *= light_factor[..., None][mask]

    # 2) Conservative chromatic adaptation. Convert environment median RGB into
    # channel gains around neutral rather than painting scene color over artwork.
    env = np.maximum(sample_median_rgb, 1.0)
    env_chroma = env / max(float(np.mean(env)), 1.0)
    gains = np.clip(env_chroma, 0.94, 1.06)
    gains = 1.0 + (gains - 1.0) * (0.55 * strength)
    rgba[..., :3][mask] *= gains[None, :]

    # 3) Tiny ambient fill color, deliberately lower than V4.1 to preserve art.
    ambient_amount = 0.020 * strength
    if ambient_amount > 0.0:
        rgba[..., :3][mask] = (
            rgba[..., :3][mask] * (1.0 - ambient_amount)
            + sample_median_rgb[None, :] * ambient_amount
        )

    # 4) Adaptive saturation matching. Only reduce saturation when the product is
    # substantially more saturated than its immediate environment.
    rgb_uint = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    hsv_product = cv2.cvtColor(rgb_uint, cv2.COLOR_RGB2HSV).astype(np.float32)
    bg_uint = np.clip(sample_rgb.reshape(-1, 1, 3), 0, 255).astype(np.uint8)
    hsv_bg = cv2.cvtColor(bg_uint, cv2.COLOR_RGB2HSV).astype(np.float32).reshape(-1, 3)

    product_sat = float(np.median(hsv_product[..., 1][mask]))
    bg_sat = float(np.median(hsv_bg[:, 1])) if len(hsv_bg) else product_sat
    excess = max(0.0, product_sat - bg_sat) / 255.0
    sat_reduction = min(0.12, excess * 0.18 * strength)
    if sat_reduction > 0.0:
        hsv_product[..., 1][mask] *= (1.0 - sat_reduction)
        rgb_adjusted = cv2.cvtColor(
            np.clip(hsv_product, 0, 255).astype(np.uint8),
            cv2.COLOR_HSV2RGB,
        ).astype(np.float32)
        rgba[..., :3][mask] = rgb_adjusted[mask]

    # 5) Low-frequency edge light wrap. Never copy sharp floor/tile texture into
    # the product edge; use the blurred environment field only.
    if wrap_strength > 0.0:
        edge_px = max(3, min(17, int(round(max(alpha.shape) * 0.0018))))
        if edge_px % 2 == 0:
            edge_px += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px, edge_px))
        dilated = cv2.dilate(alpha, kernel, iterations=1)
        eroded = cv2.erode(alpha, kernel, iterations=1)
        edge = cv2.subtract(dilated, eroded)
        edge = cv2.GaussianBlur(
            edge,
            (0, 0),
            sigmaX=max(1.0, edge_px * 0.50),
            sigmaY=max(1.0, edge_px * 0.50),
        ).astype(np.float32) / 255.0
        edge *= (alpha.astype(np.float32) / 255.0)
        edge_weight = np.clip(edge * (0.22 * wrap_strength), 0.0, 0.20)
        rgba[..., :3] = (
            rgba[..., :3] * (1.0 - edge_weight[..., None])
            + bg_blur * edge_weight[..., None]
        )

    rgba[..., :3] = np.clip(rgba[..., :3], 0, 255)
    rgba[..., 3] = np.clip(rgba[..., 3], 0, 255)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")

def contact_shadow_planar(
    alpha: Image.Image,
    strength: float = 0.18,
) -> Image.Image:
    """
    Flush-planar ambient occlusion shadow.

    Uses a tiny edge halo plus a very small downward-biased contact component.
    This avoids a floating sticker look while staying subtle enough for a mat,
    poster-on-surface, paper, or other planar object.
    """
    w, h = alpha.size
    a = np.array(alpha, dtype=np.uint8)

    dilate_px = max(2, int(round(max(w, h) * 0.0032)))
    blur_px = max(2.0, max(w, h) * 0.0044)

    kernel_size = max(3, dilate_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    dilated = cv2.dilate(a, kernel, iterations=1)
    halo = cv2.subtract(dilated, a)
    halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=blur_px, sigmaY=blur_px)

    # A tiny 1-3 px downward shift creates stronger contact on the viewer-near edge
    # without turning the mat into a floating object with a conventional drop shadow.
    shift_y = max(1, min(7, int(round(h * 0.0015))))
    shifted = np.zeros_like(a)
    shifted[shift_y:, :] = a[:-shift_y, :]
    near_contact = cv2.subtract(shifted, a)
    near_contact = cv2.GaussianBlur(
        near_contact,
        (0, 0),
        sigmaX=max(1.0, blur_px * 0.7),
        sigmaY=max(1.0, blur_px * 0.7),
    )

    s = clamp01(strength)
    combined = (
        halo.astype(np.float32) * (0.82 * s)
        + near_contact.astype(np.float32) * (1.08 * s)
    ).clip(0, 255).astype(np.uint8)

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    black.putalpha(Image.fromarray(combined, "L"))
    return Image.alpha_composite(shadow, black)


def contact_shadow_grounded(
    alpha: Image.Image,
    strength: float = 0.24,
) -> Image.Image:

    w, h = alpha.size

    a = np.array(alpha, dtype=np.uint8)
    ys, xs = np.where(a > 24)

    if len(xs) == 0:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    x1, x2 = int(xs.min()), int(xs.max())
    y2 = int(ys.max())

    obj_w = max(2, x2 - x1)
    ellipse_h = max(4, int(round(h * 0.012)))

    shadow_mask = np.zeros((h, w), dtype=np.uint8)

    center = (int((x1 + x2) / 2), min(h - 1, y2))
    axes = (max(3, int(obj_w * 0.45)), ellipse_h)

    cv2.ellipse(
        shadow_mask,
        center,
        axes,
        0,
        0,
        360,
        int(255 * clamp01(strength)),
        -1,
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (0, 0),
        sigmaX=max(3, int(w * 0.006)),
        sigmaY=max(2, int(h * 0.004)),
    )

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    black.putalpha(Image.fromarray(shadow_mask, "L"))
    return Image.alpha_composite(shadow, black)


def composite_with_support_shadow(
    background: Image.Image,
    placed_product: Image.Image,
    strategy: str,
    shadow_strength: float,
    realism_strength: float,
    edge_light_wrap: float,
) -> Image.Image:

    bg = background.convert("RGBA")
    product = harmonize_product_with_background(
        background=background,
        placed_product=placed_product,
        realism_strength=realism_strength,
        edge_light_wrap=edge_light_wrap,
    )
    alpha = product.getchannel("A")

    if strategy in {"PLANAR_SURFACE", "WALL_MOUNTED"}:
        shadow = contact_shadow_planar(alpha, strength=shadow_strength)
    else:
        shadow = contact_shadow_grounded(alpha, strength=shadow_strength)

    bg = Image.alpha_composite(bg, shadow)
    bg = Image.alpha_composite(bg, product)
    return bg.convert("RGB")


def execute_placement(
    background: Image.Image,
    crop_rgba: Image.Image,
    source_quad: np.ndarray,
    semantic: Dict[str, Any],
    plan: Dict[str, Any],
    shadow_strength: float,
    realism_strength: float,
    edge_light_wrap: float,
) -> Image.Image:

    placed = create_placed_product_layer(
        background=background,
        crop_rgba=crop_rgba,
        source_quad=source_quad,
        semantic=semantic,
        plan=plan,
    )

    return composite_with_support_shadow(
        background=background,
        placed_product=placed,
        strategy=plan["placement_strategy"],
        shadow_strength=shadow_strength,
        realism_strength=realism_strength,
        edge_light_wrap=edge_light_wrap,
    )


def create_placed_product_layer(
    background: Image.Image,
    crop_rgba: Image.Image,
    source_quad: np.ndarray,
    semantic: Dict[str, Any],
    plan: Dict[str, Any],
) -> Image.Image:

    strategy = plan["placement_strategy"]
    canvas_size = background.size

    if strategy in {"PLANAR_SURFACE", "WALL_MOUNTED"}:
        dst_quad = normalized_quad_to_pixels(
            plan["product_quad"],
            canvas_size[0],
            canvas_size[1],
        )

        placed = warp_planar_rgba(
            crop_rgba=crop_rgba,
            src_quad=source_quad,
            dst_quad_px=dst_quad,
            canvas_size=canvas_size,
        )

    else:
        placed = place_object_bbox(
            crop_rgba=crop_rgba,
            bbox_norm=plan["product_bbox"],
            canvas_size=canvas_size,
        )

    return placed


# ============================================================
# 15. REPORTING
# ============================================================

def create_thumbnail(src: Path, dst: Path):
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((300, 300), Image.LANCZOS)
        img.save(dst, "JPEG", quality=88)


def _finished_pipeline_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "pipeline_status" in df.columns:
        return df[df["pipeline_status"].astype(str) != "failed"].copy()

    legacy_success_statuses = {
        "success",
        "success_flex_unverified",
        "success_product_drift_warning",
        "success_integration_flex_unverified",
    }
    return df[df["status"].astype(str).isin(legacy_success_statuses)].copy()


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = _finished_pipeline_rows(df)

    if ok.empty:
        return pd.DataFrame()

    if "pipeline_status" not in ok.columns:
        ok["pipeline_status"] = "success"
    if "integration_status" not in ok.columns:
        ok["integration_status"] = ""
    if "final_source" not in ok.columns:
        ok["final_source"] = ""
    if "integration_rgb_delta" not in ok.columns:
        ok["integration_rgb_delta"] = ok.get("product_preservation_mae", None)

    ok["integration_rgb_delta"] = pd.to_numeric(ok["integration_rgb_delta"], errors="coerce")

    return (
        ok
        .groupby(["mode", "model_key", "model_label"], as_index=False)
        .agg(
            pipeline_success_count=(
                "pipeline_status",
                lambda s: int((s.astype(str) == "success").sum()),
            ),
            pipeline_unverified_count=(
                "pipeline_status",
                lambda s: int((s.astype(str) == "success_unverified").sum()),
            ),
            pipeline_partial_count=(
                "pipeline_status",
                lambda s: int((s.astype(str) == "partial_success").sum()),
            ),
            ai_final_count=(
                "final_source",
                lambda s: int((s.astype(str) == "ai_integration").sum()),
            ),
            deterministic_final_count=(
                "final_source",
                lambda s: int((s.astype(str) == "deterministic").sum()),
            ),
            integration_failed_count=(
                "integration_status",
                lambda s: int((s.astype(str) == "failed").sum()),
            ),
            integration_rejected_count=(
                "integration_status",
                lambda s: int((s.astype(str) == "rejected_product_drift").sum()),
            ),
            integration_rgb_delta=("integration_rgb_delta", "mean"),
            image_generation_cost_usd=("image_generation_cost_usd", "mean"),
            background_analysis_cost_usd=("background_analysis_cost_usd", "mean"),
            final_integration_cost_usd=("final_integration_cost_usd", "mean"),
            semantic_analysis_shared_cost_usd=("semantic_analysis_shared_cost_usd", "mean"),
            total_pipeline_cost_usd=("estimated_total_pipeline_cost_usd", "mean"),
            generation_latency_sec=("generation_latency_sec", "mean"),
            background_analysis_latency_sec=("background_analysis_latency_sec", "mean"),
            final_integration_latency_sec=("final_integration_latency_sec", "mean"),
            retries=("retry_count", "sum"),
        )
    )


def create_manual_review_template(df: pd.DataFrame, run_dir: Path):
    rows = []

    for _, r in df.iterrows():
        if "pipeline_status" in df.columns:
            if str(r["pipeline_status"]) == "failed":
                continue
        elif str(r["status"]) not in {
            "success",
            "success_flex_unverified",
            "success_product_drift_warning",
            "success_integration_flex_unverified",
        }:
            continue

        rows.append({
            "mode": r["mode"],
            "model_key": r["model_key"],
            "model_label": r["model_label"],
            "pipeline_status": r.get("pipeline_status", ""),
            "integration_status": r.get("integration_status", ""),
            "final_source": r.get("final_source", ""),
            "integration_rgb_delta": r.get("integration_rgb_delta", r.get("product_preservation_mae", "")),
            "warning_message": r.get("warning_message", ""),
            "semantic_category": r["semantic_product_category"],
            "placement_strategy": r["placement_strategy_used"],
            "support_surface": r["placement_support_surface"],
            "deterministic_output_file": r["deterministic_output_file"],
            "final_output_file": r["final_output_file"],

            "final_visual_quality_1to5": "",
            "final_usable_yes_no": "",
            "product_identity_ok_yes_no": "",
            "integration_notes": "",

            "product_identity_1to5": "",
            "physical_placement_1to5": "",
            "support_contact_1to5": "",
            "perspective_realism_1to5": "",
            "mask_edge_quality_1to5": "",
            "background_quality_1to5": "",
            "overall_quality_1to5": "",

            "floating_error_yes_no": "",
            "wrong_support_surface_yes_no": "",
            "usable_yes_no": "",
            "notes": "",
        })

    pd.DataFrame(rows).to_csv(
        run_dir / "manual_review_template.csv",
        index=False,
        encoding="utf-8-sig",
    )


def create_html_report(
    run_dir: Path,
    product_copy: Path,
    mask_path: Path,
    semantic: Dict[str, Any],
    semantic_usage: AnalysisUsage,
    background_description: str,
    background_source: str,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):
    thumbs = ensure_dir(run_dir / "thumbnails")

    product_thumb = thumbs / "product.jpg"
    create_thumbnail(product_copy, product_thumb)

    mask_thumb = thumbs / "mask.jpg"
    with Image.open(mask_path) as img:
        img.convert("L").save(mask_thumb, "JPEG", quality=90)

    thumb_map: Dict[str, str] = {}

    for _, r in results_df.iterrows():
        for field in ("generated_background_file", "deterministic_output_file", "final_output_file"):
            rel = r[field]
            if not rel:
                continue

            p = run_dir / rel

            if not p.exists():
                continue

            thumb = thumbs / f"{field}__{r['mode']}__{r['model_key']}.jpg"
            create_thumbnail(p, thumb)
            thumb_map[rel] = relative_path(thumb, run_dir)

    summary_rows = []
    if not summary_df.empty:
        for _, r in summary_df.iterrows():
            summary_rows.append(
                "<tr>"
                f"<td>{r['mode']}</td>"
                f"<td>{r['model_label']}</td>"
                f"<td>{int(r['pipeline_success_count'])}</td>"
                f"<td>{int(r['pipeline_unverified_count'])}</td>"
                f"<td>{int(r['pipeline_partial_count'])}</td>"
                f"<td>{int(r['ai_final_count'])}</td>"
                f"<td>{int(r['deterministic_final_count'])}</td>"
                f"<td>{int(r['integration_failed_count'])}</td>"
                f"<td>{int(r['integration_rejected_count'])}</td>"
                f"<td>{number(r['integration_rgb_delta'])}</td>"
                f"<td>{money(r['image_generation_cost_usd'])}</td>"
                f"<td>{money(r['background_analysis_cost_usd'])}</td>"
                f"<td>{money(r['final_integration_cost_usd'])}</td>"
                f"<td>{money(r['total_pipeline_cost_usd'])}</td>"
                f"<td>{number(r['generation_latency_sec'])}s</td>"
                f"<td>{number(r['background_analysis_latency_sec'])}s</td>"
                f"<td>{number(r['final_integration_latency_sec'])}s</td>"
                f"<td>{int(r['retries'])}</td>"
                "</tr>"
            )

    detail_rows = []

    for _, r in results_df.iterrows():
        bg_html = ""
        det_html = ""
        out_html = ""

        bg_rel = r["generated_background_file"]
        det_rel = r["deterministic_output_file"]
        out_rel = r["final_output_file"]

        if bg_rel in thumb_map:
            bg_html = f'<a href="{bg_rel}" target="_blank"><img src="{thumb_map[bg_rel]}"></a>'

        if det_rel in thumb_map:
            det_html = f'<a href="{det_rel}" target="_blank"><img src="{thumb_map[det_rel]}"></a>'

        if out_rel in thumb_map:
            out_html = f'<a href="{out_rel}" target="_blank"><img src="{thumb_map[out_rel]}"></a>'

        integration_rgb_delta = r.get("integration_rgb_delta", r.get("product_preservation_mae", ""))

        detail_rows.append(
            "<tr>"
            f"<td>{r['mode']}</td>"
            f"<td>{r['model_label']}</td>"
            f"<td>{bg_html}</td>"
            f"<td>{det_html}</td>"
            f"<td>{out_html}</td>"
            f"<td>{r['placement_strategy_used']}</td>"
            f"<td>{r['placement_support_surface']}</td>"
            f"<td>{number(r['placement_confidence'], 3)}</td>"
            f"<td>{r['traffic_type']}</td>"
            f"<td>{r['integration_traffic_type']}</td>"
            f"<td>{number(integration_rgb_delta)}</td>"
            f"<td>{money(r['estimated_total_pipeline_cost_usd'])}</td>"
            f"<td>{r.get('pipeline_status', '')}</td>"
            f"<td>{r.get('integration_status', '')}</td>"
            f"<td>{r.get('final_source', '')}</td>"
            f"<td>{r.get('warning_message', '')}</td>"
            f"<td>{r.get('error_message', '')}</td>"
            f"<td>{r['status']}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Task 4 V4.1 — Semantic Product Placement</title>
<style>
body {{ font-family: Arial,sans-serif; margin:24px; color:#222; line-height:1.45; }}
h1,h2 {{ color:#17365D; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 28px; }}
th,td {{ border:1px solid #ddd; padding:7px; vertical-align:top; }}
th {{ background:#17365D; color:white; }}
img {{ max-width:210px; display:block; }}
pre {{ white-space:pre-wrap; background:#f5f5f5; padding:10px; }}
.note {{ background:#fff2cc; padding:12px; margin:12px 0; }}
.info {{ background:#eef5fb; padding:12px; margin:12px 0; }}
</style>
</head>
<body>

<h1>Task 4 V4.1 — Semantic Product Placement + Photoreal Integration</h1>

<div class="note">
No product category is hardcoded. The system first analyzes the product,
then selects a physical placement strategy and support surface.
Integration RGB Delta is telemetry only. Relighting, color integration, shadow,
and edge blending may increase it while improving final visual realism. Manual
visual review remains the primary quality signal for usable final output.
</div>

<h2>Input</h2>
<a href="{relative_path(product_copy, run_dir)}" target="_blank">
<img src="{relative_path(product_thumb, run_dir)}">
</a>

<h2>Semantic Product Analysis</h2>
<pre>{json.dumps(semantic, ensure_ascii=False, indent=2)}</pre>

<div class="info">
Semantic analysis latency: {number(semantic_usage.latency_sec)}s<br>
Semantic analysis cost: {money(semantic_usage.estimated_cost_usd)}
</div>

<h2>Segmentation Mask</h2>
<a href="{relative_path(mask_path, run_dir)}" target="_blank">
<img src="{relative_path(mask_thumb, run_dir)}">
</a>

<h2>Requested Background</h2>
<div class="info">Background source: {background_source}</div>
<pre>{background_description}</pre>

<h2>Cost / Latency Summary</h2>
<table>
<thead>
<tr>
<th>Mode</th>
<th>Model</th>
<th>Pipeline success</th>
<th>Unverified</th>
<th>Partial</th>
<th>AI final</th>
<th>Deterministic final</th>
<th>Integration failed</th>
<th>Integration rejected</th>
<th>Integration RGB Delta</th>
<th>Image cost</th>
<th>Placement-analysis cost</th>
<th>Final-integration cost</th>
<th>Total pipeline cost</th>
<th>Image latency</th>
<th>Placement-analysis latency</th>
<th>Final-integration latency</th>
<th>Retries</th>
</tr>
</thead>
<tbody>
{''.join(summary_rows)}
</tbody>
</table>

<h2>Outputs</h2>
<table>
<thead>
<tr>
<th>Mode</th>
<th>Model</th>
<th>Generated background</th>
<th>Deterministic composite</th>
<th>Final composite</th>
<th>Placement</th>
<th>Support</th>
<th>Placement confidence</th>
<th>Traffic type</th>
<th>Integration traffic</th>
<th>Integration RGB Delta</th>
<th>Total pipeline cost</th>
<th>Pipeline status</th>
<th>Integration status</th>
<th>Final source</th>
<th>Warning</th>
<th>Error</th>
<th>Status detail</th>
</tr>
</thead>
<tbody>
{''.join(detail_rows)}
</tbody>
</table>

</body>
</html>
"""

    (run_dir / "report.html").write_text(html, encoding="utf-8")


# ============================================================
# 16. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 4 V4.1: functional-anchor-aware product placement with geometry guards, deterministic harmonization, and photoreal integration."
    )

    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--product-file", default="")
    parser.add_argument("--mask-file", default="")
    parser.add_argument("--background", default="")
    parser.add_argument("--background-file", default="")
    parser.add_argument(
        "--background-mode",
        choices=["auto", "dataset", "prompt"],
        default="auto",
        help="auto creates a product-aware background brief; dataset uses dataset/background.txt; prompt asks interactively.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))

    parser.add_argument("--analysis-model", default=os.getenv("SEMANTIC_ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL))

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["standard", "flex"],
        default=["flex"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["nb2", "pro"],
        default=["pro"],
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--jpeg-quality", type=int, default=95)

    parser.add_argument(
        "--semantic-confidence-min",
        type=float,
        default=0.60,
        help="Below this, script warns and asks before continuing unless --yes.",
    )

    parser.add_argument(
        "--placement-confidence-min",
        type=float,
        default=0.45,
        help="Below this, fallback placement may be used.",
    )

    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=0.38,
        help="Contact-shadow strength. Planar products usually need 0.25-0.40 to avoid a sticker look.",
    )

    parser.add_argument(
        "--realism-strength",
        type=float,
        default=0.70,
        help="How strongly the placed product inherits local background lighting/tint (0 disables).",
    )

    parser.add_argument(
        "--edge-light-wrap",
        type=float,
        default=0.85,
        help="How much local background color is blended into the product edge band (0 disables).",
    )

    parser.add_argument(
        "--final-integration",
        choices=["on", "off"],
        default="on",
        help="Run a final image-model relight/blend pass for maximum photorealism.",
    )

    parser.add_argument(
        "--product-preservation-mae-max",
        type=float,
        default=100.0,
        help="Integration RGB Delta warning threshold after final integration.",
    )

    parser.add_argument(
        "--product-preservation-policy",
        choices=["warn", "reject", "off"],
        default="warn",
        help="warn keeps the photoreal integration but records drift; reject falls back to deterministic output.",
    )

    parser.add_argument(
        "--min-planar-area-ratio",
        type=float,
        default=0.10,
        help="Minimum projected canvas-area fraction for planar ecommerce hero products.",
    )
    parser.add_argument(
        "--max-planar-area-ratio",
        type=float,
        default=0.32,
        help="Maximum projected canvas-area fraction for planar products.",
    )
    parser.add_argument(
        "--orientation-preservation",
        type=float,
        default=0.90,
        help="How strongly target geometry must preserve the source dominant aspect ratio (0.5..1.0).",
    )
    parser.add_argument(
        "--minimum-dominant-ratio",
        type=float,
        default=1.20,
        help="Minimum visible dominant-axis ratio for HORIZONTAL/VERTICAL products.",
    )
    parser.add_argument(
        "--placement-canvas-margin",
        type=float,
        default=0.025,
        help="Safe normalized margin used while expanding/correcting planar quads.",
    )
    parser.add_argument(
        "--perspective-guard-strength",
        type=float,
        default=0.85,
        help="How strongly floor-plane perspective hints regularize planar quads (0..1).",
    )
    parser.add_argument(
        "--max-near-far-ratio",
        type=float,
        default=1.35,
        help="Maximum allowed near/far edge scale ratio for floor-planar perspective.",
    )
    parser.add_argument(
        "--max-opposite-edge-angle-delta-deg",
        type=float,
        default=12.0,
        help="Maximum tolerated local angle difference between opposite width edges before regularization.",
    )

    parser.add_argument(
        "--context-guard-strength",
        type=float,
        default=0.95,
        help="How strongly detected functional anchors influence product position/alignment (0..1).",
    )
    parser.add_argument(
        "--functional-anchor-confidence-min",
        type=float,
        default=0.55,
        help="Minimum detected anchor confidence before anchor-aware placement is enforced.",
    )
    parser.add_argument(
        "--context-score-min",
        type=float,
        default=0.60,
        help="Minimum planner context score before contextual replanning is attempted.",
    )
    parser.add_argument(
        "--context-replan-attempts",
        type=int,
        default=1,
        help="Additional placement-analysis attempts when functional context is important but weak.",
    )
    parser.add_argument(
        "--max-anchor-shift-ratio",
        type=float,
        default=0.40,
        help="Maximum contextual placement shift as a fraction of the shorter canvas side.",
    )
    parser.add_argument(
        "--max-anchor-rotation-deg",
        type=float,
        default=20.0,
        help="Maximum functional-anchor alignment rotation applied to planar products.",
    )
    parser.add_argument(
        "--anchor-gap-ratio",
        type=float,
        default=0.012,
        help="Base gap between product and functional anchor as a fraction of the shorter canvas side.",
    )
    parser.add_argument(
        "--mask-contract-ratio",
        type=float,
        default=0.0015,
        help="Tiny inward alpha contraction as fraction of the shorter source dimension; suppresses segmentation fringe.",
    )
    parser.add_argument(
        "--mask-feather-ratio",
        type=float,
        default=0.0008,
        help="Alpha feather sigma as fraction of the shorter source dimension.",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")

    return parser.parse_args()


# ============================================================
# 17. MAIN
# ============================================================

def main():
    args = parse_args()

    if not args.project:
        print()
        print("ERROR: GOOGLE_CLOUD_PROJECT is empty.")
        print(f".env expected at: {ENV_PATH}")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    product_path = resolve_product_file(dataset_dir, args.product_file)
    explicit_mask = resolve_mask_file(args.mask_file)
    background_override = resolve_background_override(args)

    product_img = load_rgb(product_path)
    source_w, source_h = product_img.size
    aspect_ratio = nearest_aspect_ratio(source_w, source_h)

    # Analysis always uses standard routing by default.
    analysis_client = create_client(
        project=args.project,
        location=args.location,
        mode="standard",
    )

    print()
    print("Analyzing product semantics...")

    semantic_raw, semantic_usage = call_json_vision(
        client=analysis_client,
        model_id=args.analysis_model,
        image=product_img,
        prompt=semantic_analysis_prompt(),
        temperature=0.1,
    )

    semantic = normalize_semantic_analysis(semantic_raw)

    print(json.dumps(semantic, ensure_ascii=False, indent=2))
    print(
        f"Semantic confidence: {semantic['confidence']:.3f}"
        f" | latency {number(semantic_usage.latency_sec)}s"
        f" | cost {money(semantic_usage.estimated_cost_usd)}"
    )

    if semantic["confidence"] < args.semantic_confidence_min:
        print()
        print(
            f"WARNING: semantic confidence {semantic['confidence']:.3f} "
            f"is below threshold {args.semantic_confidence_min:.3f}."
        )

        if not args.yes:
            answer = input("Continue anyway? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Cancelled.")
                return

    if background_override is not None:
        background_description = background_override
        background_source = "explicit"
    elif args.background_mode == "dataset":
        background_description = resolve_dataset_background_description(args)
        if background_description is None:
            raise FileNotFoundError(
                "background.txt was requested with --background-mode dataset, "
                "but no non-empty dataset/background.txt was found."
            )
        background_source = "dataset/background.txt"
    elif args.background_mode == "prompt":
        background_description = prompt_background_description()
        background_source = "interactive prompt"
    else:
        background_description = auto_background_description(semantic)
        background_source = "auto product-aware brief"

    print()
    print(f"Background source   : {background_source}")
    print("Background brief:")
    print(background_description)

    print()
    print("Segmenting exact product...")

    mask, segmentation_method = segment_product(
        product_path=product_path,
        img=product_img,
        explicit_mask=explicit_mask,
        mask_contract_ratio=args.mask_contract_ratio,
        mask_feather_ratio=args.mask_feather_ratio,
    )

    coverage = mask_coverage(mask)

    product_rgba = rgba_from_product_and_mask(product_img, mask)
    crop_rgba, source_bbox = crop_rgba_to_mask(product_rgba, mask)
    source_quad = estimate_source_quad(mask, source_bbox)
    product_geometry = source_product_geometry(crop_rgba, source_quad, semantic)

    modes = list(dict.fromkeys(args.modes))
    models = list(dict.fromkeys(args.models))
    request_count = len(modes) * len(models)
    image_request_count = request_count * (2 if args.final_integration == "on" else 1)

    preview = preview_output_only_cost(modes, models)
    if args.final_integration == "on":
        preview *= 2

    print()
    print("=" * 80)
    print("TASK 4 V4.1 — SEMANTIC PRODUCT PLACEMENT")
    print("=" * 80)
    print(f"Product             : {product_path}")
    print(f"Source size         : {source_w}x{source_h}")
    print(f"Detected category   : {semantic['product_category']}")
    print(f"Physical form       : {semantic['physical_form']}")
    print(f"Natural support     : {semantic['natural_support']}")
    print(f"Placement strategy  : {semantic['placement_strategy']}")
    print(f"Semantic confidence : {semantic['confidence']:.3f}")
    print(f"Segmentation        : {segmentation_method}")
    print(f"Mask coverage       : {coverage:.4f}")
    print(
        f"Product crop ratio  : {product_geometry['crop_aspect_ratio_w_over_h']:.3f} "
        f"(W/H) | orientation={semantic['natural_orientation']}"
    )
    print(f"Modes               : {', '.join(modes)}")
    print(f"Models              : {', '.join(models)}")
    print(f"Placement cases     : {request_count}")
    print(f"Image requests      : {image_request_count}")
    print(f"Final integration   : {args.final_integration}")
    print(f"Preservation policy : {args.product_preservation_policy}")
    print(f"4K output-only preview: ~${preview:.6f}")
    print("=" * 80)

    if args.dry_run:
        print("DRY RUN: semantic analysis was performed, but image generation was not.")
        return

    if not args.yes:
        answer = input("Proceed with image generation? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root).expanduser().resolve() / timestamp)

    inputs_dir = ensure_dir(run_dir / "inputs")
    intermediates_dir = ensure_dir(run_dir / "intermediates")
    backgrounds_dir = ensure_dir(run_dir / "backgrounds")
    outputs_dir = ensure_dir(run_dir / "outputs")
    analysis_dir = ensure_dir(run_dir / "analysis")

    product_copy = inputs_dir / product_path.name
    shutil.copy2(product_path, product_copy)

    mask_path = intermediates_dir / "product_mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    cutout_path = intermediates_dir / "product_cutout.png"
    crop_rgba.save(cutout_path)

    (analysis_dir / "semantic_product_analysis.json").write_text(
        json.dumps(semantic, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_dir / "background.txt").write_text(background_description, encoding="utf-8")

    bg_prompt = build_background_prompt(background_description, semantic)

    (run_dir / "background_prompt.txt").write_text(bg_prompt, encoding="utf-8")

    clients = {
        mode: create_client(
            project=args.project,
            location=args.location,
            mode=mode,
        )
        for mode in modes
    }

    image_config = build_image_generation_config(
        aspect_ratio=aspect_ratio,
        temperature=args.temperature,
        jpeg_quality=args.jpeg_quality,
    )

    results: List[ResultRow] = []
    current = 0

    for mode in modes:
        for model_key in models:
            current += 1
            model = IMAGE_MODELS[model_key]

            print()
            print(f"[{current}/{request_count}] {mode.upper()} | {model['label']}")

            generated = generate_background_with_retry(
                client=clients[mode],
                mode=mode,
                model_id=model["id"],
                prompt=bg_prompt,
                config=image_config,
            )

            if not generated["ok"]:
                err = generated["error"]
                msg = f"{type(err).__name__}: {err}" if err is not None else "Unknown error"
                print(f"  ERROR: {msg}")

                results.append(
                    ResultRow(
                        mode=mode,
                        model_key=model_key,
                        model_label=model["label"],
                        model_id=model["id"],
                        product_file=product_path.name,
                        source_width=source_w,
                        source_height=source_h,
                        background_width=None,
                        background_height=None,
                        semantic_product_category=semantic["product_category"],
                        semantic_physical_form=semantic["physical_form"],
                        semantic_natural_support=semantic["natural_support"],
                        semantic_natural_orientation=semantic["natural_orientation"],
                        semantic_placement_strategy=semantic["placement_strategy"],
                        semantic_confidence=semantic["confidence"],
                        segmentation_method=segmentation_method,
                        mask_coverage_ratio=coverage,
                        placement_strategy_used="",
                        placement_confidence=0.0,
                        placement_support_surface="",
                        placement_plan_json="",
                        generated_background_file="",
                        deterministic_output_file="",
                        integration_output_file="",
                        final_output_file="",
                        pipeline_status="failed",
                        integration_status="not_run",
                        final_source="none",
                        status="error",
                        traffic_type="",
                        integration_traffic_type="",
                        flex_verified=None,
                        integration_flex_verified=None,
                        pricing_verified=False,
                        attempts_used=generated["attempts_used"],
                        retry_count=generated["retry_count"],
                        retry_wait_sec=generated["retry_wait_sec"],
                        generation_latency_sec=generated["end_to_end_latency_sec"],
                        background_analysis_latency_sec=None,
                        final_integration_latency_sec=None,
                        prompt_tokens=None,
                        thinking_tokens=None,
                        output_image_tokens=None,
                        integration_prompt_tokens=None,
                        integration_thinking_tokens=None,
                        integration_output_image_tokens=None,
                        product_preservation_mae=None,
                        integration_rgb_delta=None,
                        image_generation_cost_usd=None,
                        background_analysis_cost_usd=None,
                        final_integration_cost_usd=None,
                        semantic_analysis_shared_cost_usd=semantic_usage.estimated_cost_usd,
                        estimated_total_pipeline_cost_usd=None,
                        warning_message="",
                        error_message=msg,
                    )
                )
                continue

            bg_mode_dir = ensure_dir(backgrounds_dir / mode)
            out_mode_dir = ensure_dir(outputs_dir / mode)

            bg_path = bg_mode_dir / f"{model_key}_background_4k.jpg"
            bg_path.write_bytes(generated["image_bytes"])

            with Image.open(bg_path) as bg:
                background_img = bg.convert("RGB")

            bg_w, bg_h = background_img.size

            print(f"  Generated background: {bg_w}x{bg_h}")
            print("  Analyzing support surface / physical placement...")

            planning_prompt = placement_analysis_prompt(semantic, product_geometry)
            plan_raw, first_plan_usage = call_json_vision(
                client=analysis_client,
                model_id=args.analysis_model,
                image=background_img,
                prompt=planning_prompt,
                temperature=0.1,
            )

            plan = normalize_placement_plan(
                raw=plan_raw,
                semantic_strategy=semantic["placement_strategy"],
            )
            plan_usages: List[AnalysisUsage] = [first_plan_usage]

            # Contextual replan: cheap vision-analysis retry when the scene contains
            # a functional anchor but the first planner did not use it well enough.
            for replan_idx in range(max(0, int(args.context_replan_attempts))):
                if placement_context_is_acceptable(
                    plan,
                    semantic,
                    anchor_confidence_min=args.functional_anchor_confidence_min,
                    context_score_min=args.context_score_min,
                ):
                    break

                print(
                    f"  Context weak (score={plan.get('context_score', 0.0):.3f}, "
                    f"anchor_conf={plan.get('functional_anchor', {}).get('confidence', 0.0):.3f}); "
                    f"contextual replan {replan_idx + 1}..."
                )
                reinforced_prompt = planning_prompt + """

CRITICAL REPLAN:
The previous plan was not contextually convincing. Re-inspect the image for the
functional anchor specified in product semantics. If the anchor is visible, place
the product where a real customer would actually use it relative to that anchor,
not in a merely empty region. Return accurate anchor_bbox/anchor_line/anchor_point,
placement_side, alignment, and context_score. Do not invent an anchor if absent.
"""
                raw2, usage2 = call_json_vision(
                    client=analysis_client,
                    model_id=args.analysis_model,
                    image=background_img,
                    prompt=reinforced_prompt,
                    temperature=0.05,
                )
                plan2 = normalize_placement_plan(
                    raw=raw2,
                    semantic_strategy=semantic["placement_strategy"],
                )
                plan_usages.append(usage2)
                if placement_plan_quality_score(plan2, semantic) > placement_plan_quality_score(plan, semantic):
                    plan = plan2

            plan_usage = aggregate_analysis_usage(plan_usages)

            # Functional placement has first-class status. Low generic placement
            # confidence does not force a center-of-canvas fallback when a strong
            # functional anchor/context plan is available.
            anchor_conf = float(plan.get("functional_anchor", {}).get("confidence", 0.0))
            context_score = float(plan.get("context_score", 0.0))
            functional_priority = float(
                normalize_functional_context(semantic.get("functional_context")).get("context_priority", 0.0)
            )
            effective_placement_conf = max(
                float(plan.get("placement_confidence", 0.0)),
                anchor_conf * context_score * functional_priority,
            )

            if effective_placement_conf < args.placement_confidence_min:
                print(
                    f"  Effective placement confidence {effective_placement_conf:.3f} "
                    f"< {args.placement_confidence_min:.3f}; using strategy fallback."
                )
                fallback = fallback_plan_for_strategy(plan["placement_strategy"])
                # Keep detected anchor metadata for audit even if geometry falls back.
                fallback["functional_anchor"] = plan.get("functional_anchor", fallback["functional_anchor"])
                fallback["context_score"] = plan.get("context_score", 0.0)
                plan = fallback

            plan = apply_functional_context_guard(
                plan=plan,
                semantic=semantic,
                canvas_size=background_img.size,
                strength=args.context_guard_strength,
                anchor_confidence_min=args.functional_anchor_confidence_min,
                max_anchor_shift_ratio=args.max_anchor_shift_ratio,
                max_anchor_rotation_deg=args.max_anchor_rotation_deg,
                anchor_gap_ratio=args.anchor_gap_ratio,
                canvas_margin=args.placement_canvas_margin,
            )

            fguard = plan.get("functional_guard", {})
            if fguard.get("enabled"):
                print(
                    f"  Functional guard: anchor={fguard.get('anchor_type')} "
                    f"| side={fguard.get('placement_side', 'NONE')} "
                    f"| shift={fguard.get('shift_px', 0.0):.1f}px "
                    f"| changes={','.join(fguard.get('changes', [])) or 'none'}"
                )
            else:
                print(
                    f"  Functional guard: inactive "
                    f"({fguard.get('reason', 'no high-confidence functional constraint')})"
                )

            plan = stabilize_placement_plan(
                plan=plan,
                semantic=semantic,
                product_geometry=product_geometry,
                canvas_size=background_img.size,
                min_planar_area_ratio=args.min_planar_area_ratio,
                max_planar_area_ratio=args.max_planar_area_ratio,
                orientation_preservation=args.orientation_preservation,
                minimum_dominant_ratio=args.minimum_dominant_ratio,
                canvas_margin=args.placement_canvas_margin,
                perspective_guard_strength=args.perspective_guard_strength,
                max_near_far_ratio=args.max_near_far_ratio,
                max_opposite_edge_angle_delta_deg=args.max_opposite_edge_angle_delta_deg,
            )

            guard = plan.get("geometry_guard", {})
            if guard.get("enabled"):
                print(
                    f"  Geometry guard: ratio "
                    f"{guard.get('before_width_height_ratio', 0):.3f} -> "
                    f"{guard.get('after_width_height_ratio', 0):.3f} "
                    f"| area {guard.get('before_area_ratio', 0):.3f} -> "
                    f"{guard.get('after_area_ratio', 0):.3f}"
                )
                if guard.get("changes"):
                    print(f"  Geometry changes: {', '.join(guard['changes'])}")

                pguard = guard.get("perspective_guard", {})
                if pguard.get("enabled"):
                    print(
                        f"  Perspective guard: near/far "
                        f"{pguard.get('before_near_far_ratio', 1.0):.3f} -> "
                        f"{pguard.get('after_near_far_ratio', 1.0):.3f} "
                        f"| hint={pguard.get('near_far_scale_ratio_hint', 1.0):.3f} "
                        f"| conf={pguard.get('surface_confidence', 0.0):.3f}"
                    )

            print(
                f"  Placement: {plan['placement_strategy']} "
                f"| support={plan['support_surface_type']} "
                f"| confidence={plan['placement_confidence']:.3f} "
                f"| context={plan.get('context_score', 0.0):.3f} "
                f"| anchor={plan.get('functional_anchor', {}).get('anchor_type', 'NONE')}"
            )

            plan_path = analysis_dir / f"{mode}__{model_key}__placement_plan.json"
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            save_placement_debug_overlay(
                background=background_img,
                plan=plan,
                output_path=analysis_dir / f"{mode}__{model_key}__placement_overlay.jpg",
            )

            placed_layer = create_placed_product_layer(
                background=background_img,
                crop_rgba=crop_rgba,
                source_quad=source_quad,
                semantic=semantic,
                plan=plan,
            )

            deterministic_final = composite_with_support_shadow(
                background=background_img,
                placed_product=placed_layer,
                strategy=plan["placement_strategy"],
                shadow_strength=args.shadow_strength,
                realism_strength=args.realism_strength,
                edge_light_wrap=args.edge_light_wrap,
            )

            det_path = out_mode_dir / f"{model_key}_deterministic_composite.png"
            out_path = out_mode_dir / f"{model_key}_semantic_strict_output.png"
            deterministic_final.save(det_path, "PNG", optimize=True)
            deterministic_final.save(out_path, "PNG", optimize=True)

            final = deterministic_final
            integration_result: Optional[Dict[str, Any]] = None
            integration_cost: Optional[float] = None
            integration_prompt_tokens: Optional[int] = None
            integration_thinking_tokens: Optional[int] = None
            integration_output_image_tokens: Optional[int] = None
            integration_latency: Optional[float] = None
            integration_raw_tt = ""
            integration_norm_tt = ""
            integration_flex_verified: Optional[bool] = None
            integration_output_file = ""
            integration_error = ""
            integration_warning = ""
            integration_rejected_for_drift = False
            integration_drift_warning = False
            product_preservation_mae: Optional[float] = None

            if args.final_integration == "on":
                print("  Running final photoreal integration / relight pass...")
                integration_prompt = build_final_integration_prompt(
                    background_description=background_description,
                    semantic=semantic,
                    plan=plan,
                )
                integration_result = generate_final_integration_with_retry(
                    client=clients[mode],
                    mode=mode,
                    model_id=model["id"],
                    prompt=integration_prompt,
                    product_reference=product_img,
                    clean_background=background_img,
                    deterministic_composite=deterministic_final,
                    product_alpha_mask=alpha_mask_preview(placed_layer),
                    config=image_config,
                )

                integration_latency = integration_result["end_to_end_latency_sec"]

                if integration_result["ok"]:
                    with Image.open(io.BytesIO(integration_result["image_bytes"])) as integrated_img:
                        final = integrated_img.convert("RGB")
                    if final.size != background_img.size:
                        final = final.resize(background_img.size, Image.LANCZOS)

                    product_preservation_mae = masked_rgb_mae(
                        reference=deterministic_final,
                        candidate=final,
                        alpha=placed_layer.getchannel("A"),
                    )

                    product_drift_over_limit = (
                        product_preservation_mae is not None
                        and product_preservation_mae > args.product_preservation_mae_max
                    )

                    if product_drift_over_limit and args.product_preservation_policy == "reject":
                        rejected_path = out_mode_dir / f"{model_key}_rejected_integration.png"
                        final.save(rejected_path, "PNG", optimize=True)
                        final = deterministic_final
                        deterministic_final.save(out_path, "PNG", optimize=True)
                        integration_output_file = relative_path(rejected_path, run_dir)
                        integration_rejected_for_drift = True
                        integration_warning = (
                            "Final integration rejected by policy: integration RGB delta "
                            f"{product_preservation_mae:.2f} > "
                            f"{args.product_preservation_mae_max:.2f}"
                        )
                    else:
                        final.save(out_path, "PNG", optimize=True)
                        integration_output_file = relative_path(out_path, run_dir)
                        if product_drift_over_limit and args.product_preservation_policy == "warn":
                            integration_drift_warning = True
                            integration_warning = (
                                "Final integration RGB delta warning: "
                                f"{product_preservation_mae:.2f} > "
                                f"{args.product_preservation_mae_max:.2f}; "
                                "kept because policy=warn."
                            )

                    integration_usage = integration_result["usage"]
                    integration_prompt_tokens = integration_usage.get("prompt_tokens")
                    integration_thinking_tokens = integration_usage.get("thinking_tokens")
                    integration_output_image_tokens = integration_result["output_image_tokens"]
                    integration_raw_tt = integration_result["traffic_type_raw"]
                    integration_norm_tt = integration_result["traffic_type_normalized"]

                    if mode == "flex":
                        integration_flex_verified = integration_norm_tt == EXPECTED_FLEX_TRAFFIC_TYPE

                    integration_cost = estimate_image_cost(
                        mode=mode,
                        model_key=model_key,
                        prompt_tokens=integration_prompt_tokens,
                        thinking_tokens=integration_thinking_tokens,
                        output_image_tokens=integration_output_image_tokens,
                        pricing_verified=(
                            integration_flex_verified
                            if mode == "flex"
                            else True
                        ),
                    )
                else:
                    err = integration_result["error"]
                    integration_error = (
                        f"Final integration failed: {type(err).__name__}: {err}"
                        if err is not None
                        else "Final integration failed: unknown error"
                    )

            usage = generated["usage"]
            prompt_tokens = usage.get("prompt_tokens")
            thinking_tokens = usage.get("thinking_tokens")
            output_image_tokens = generated["output_image_tokens"]

            raw_tt = generated["traffic_type_raw"]
            norm_tt = generated["traffic_type_normalized"]

            if mode == "flex":
                flex_verified = norm_tt == EXPECTED_FLEX_TRAFFIC_TYPE
                pricing_verified = flex_verified
                if args.final_integration == "on" and integration_result is not None and integration_result["ok"]:
                    pricing_verified = pricing_verified and bool(integration_flex_verified)
            else:
                flex_verified = None
                pricing_verified = True

            image_cost = estimate_image_cost(
                mode=mode,
                model_key=model_key,
                prompt_tokens=prompt_tokens,
                thinking_tokens=thinking_tokens,
                output_image_tokens=output_image_tokens,
                pricing_verified=pricing_verified,
            )

            semantic_shared_cost = semantic_usage.estimated_cost_usd
            placement_cost = plan_usage.estimated_cost_usd

            total_pipeline_cost = None
            if image_cost is not None:
                total_pipeline_cost = (
                    image_cost
                    + (semantic_shared_cost or 0.0)
                    + (placement_cost or 0.0)
                    + (integration_cost or 0.0)
                )

            pipeline_status = "success"
            integration_status = "not_run"
            final_source = "deterministic"
            status = "success"

            if mode == "flex" and not flex_verified:
                pipeline_status = "success_unverified"
                status = "success_flex_unverified"

            if args.final_integration == "on":
                if integration_result is not None and integration_result["ok"]:
                    integration_status = "accepted"
                    final_source = "ai_integration"

                    if integration_rejected_for_drift:
                        pipeline_status = "partial_success"
                        integration_status = "rejected_product_drift"
                        final_source = "deterministic"
                        status = "integration_rejected_deterministic_fallback"
                    elif integration_drift_warning:
                        integration_status = "accepted_with_drift_warning"
                        status = "success_product_drift_warning"
                        if mode == "flex" and not integration_flex_verified:
                            pipeline_status = "success_unverified"
                            integration_status = "accepted_with_drift_warning_routing_unverified"
                    elif mode == "flex" and not integration_flex_verified:
                        pipeline_status = "success_unverified"
                        integration_status = "accepted_routing_unverified"
                        status = "success_integration_flex_unverified"
                else:
                    pipeline_status = "partial_success"
                    integration_status = "failed"
                    final_source = "deterministic"
                    status = "integration_failed_deterministic_fallback"

            print(
                f"  FINAL PNG {final.size[0]}x{final.size[1]}"
                f" | image cost {money(image_cost)}"
                f" | plan cost {money(placement_cost)}"
                f" | integration cost {money(integration_cost)}"
                f" | total pipeline {money(total_pipeline_cost)}"
                f" | final source {final_source}"
                f" | integration {integration_status}"
            )

            results.append(
                ResultRow(
                    mode=mode,
                    model_key=model_key,
                    model_label=model["label"],
                    model_id=model["id"],
                    product_file=product_path.name,
                    source_width=source_w,
                    source_height=source_h,
                    background_width=bg_w,
                    background_height=bg_h,
                    semantic_product_category=semantic["product_category"],
                    semantic_physical_form=semantic["physical_form"],
                    semantic_natural_support=semantic["natural_support"],
                    semantic_natural_orientation=semantic["natural_orientation"],
                    semantic_placement_strategy=semantic["placement_strategy"],
                    semantic_confidence=semantic["confidence"],
                    segmentation_method=segmentation_method,
                    mask_coverage_ratio=coverage,
                    placement_strategy_used=plan["placement_strategy"],
                    placement_confidence=plan["placement_confidence"],
                    placement_support_surface=plan["support_surface_type"],
                    placement_plan_json=json.dumps(plan, ensure_ascii=False),
                    generated_background_file=relative_path(bg_path, run_dir),
                    deterministic_output_file=relative_path(det_path, run_dir),
                    integration_output_file=integration_output_file,
                    final_output_file=relative_path(out_path, run_dir),
                    pipeline_status=pipeline_status,
                    integration_status=integration_status,
                    final_source=final_source,
                    status=status,
                    traffic_type=norm_tt or raw_tt,
                    integration_traffic_type=integration_norm_tt or integration_raw_tt,
                    flex_verified=flex_verified,
                    integration_flex_verified=integration_flex_verified,
                    pricing_verified=pricing_verified,
                    attempts_used=generated["attempts_used"],
                    retry_count=generated["retry_count"],
                    retry_wait_sec=generated["retry_wait_sec"],
                    generation_latency_sec=generated["end_to_end_latency_sec"],
                    background_analysis_latency_sec=plan_usage.latency_sec,
                    final_integration_latency_sec=integration_latency,
                    prompt_tokens=prompt_tokens,
                    thinking_tokens=thinking_tokens,
                    output_image_tokens=output_image_tokens,
                    integration_prompt_tokens=integration_prompt_tokens,
                    integration_thinking_tokens=integration_thinking_tokens,
                    integration_output_image_tokens=integration_output_image_tokens,
                    product_preservation_mae=product_preservation_mae,
                    integration_rgb_delta=product_preservation_mae,
                    image_generation_cost_usd=image_cost,
                    background_analysis_cost_usd=placement_cost,
                    final_integration_cost_usd=integration_cost,
                    semantic_analysis_shared_cost_usd=semantic_shared_cost,
                    estimated_total_pipeline_cost_usd=total_pipeline_cost,
                    warning_message=integration_warning,
                    error_message=integration_error,
                )
            )

            pd.DataFrame([asdict(r) for r in results]).to_csv(
                run_dir / "results.csv",
                index=False,
                encoding="utf-8-sig",
            )

    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df.to_csv(run_dir / "results.csv", index=False, encoding="utf-8-sig")

    summary_df = create_summary(results_df)
    summary_df.to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    create_manual_review_template(results_df, run_dir)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "backend": "Google Cloud / Gemini Enterprise Agent Platform",
        "project": args.project,
        "location": args.location,
        "task": "Semantic Product Placement V4.1 — Functional Anchor + Perspective Guard",
        "analysis_model": args.analysis_model,
        "product_file": str(product_path),
        "background_source": background_source,
        "background_description": background_description,
        "semantic_analysis": semantic,
        "semantic_analysis_usage": asdict(semantic_usage),
        "product_geometry": product_geometry,
        "geometry_guard_settings": {
            "min_planar_area_ratio": args.min_planar_area_ratio,
            "max_planar_area_ratio": args.max_planar_area_ratio,
            "orientation_preservation": args.orientation_preservation,
            "minimum_dominant_ratio": args.minimum_dominant_ratio,
            "placement_canvas_margin": args.placement_canvas_margin,
            "perspective_guard_strength": args.perspective_guard_strength,
            "max_near_far_ratio": args.max_near_far_ratio,
            "max_opposite_edge_angle_delta_deg": args.max_opposite_edge_angle_delta_deg,
        },
        "contextual_placement_settings": {
            "context_guard_strength": args.context_guard_strength,
            "functional_anchor_confidence_min": args.functional_anchor_confidence_min,
            "context_score_min": args.context_score_min,
            "context_replan_attempts": args.context_replan_attempts,
            "max_anchor_shift_ratio": args.max_anchor_shift_ratio,
            "max_anchor_rotation_deg": args.max_anchor_rotation_deg,
            "anchor_gap_ratio": args.anchor_gap_ratio,
        },
        "mask_cleanup_settings": {
            "mask_contract_ratio": args.mask_contract_ratio,
            "mask_feather_ratio": args.mask_feather_ratio,
        },
        "compositing_settings": {
            "shadow_strength": args.shadow_strength,
            "realism_strength": args.realism_strength,
            "edge_light_wrap": args.edge_light_wrap,
            "final_integration": args.final_integration,
            "product_preservation_mae_max": args.product_preservation_mae_max,
            "product_preservation_policy": args.product_preservation_policy,
        },
        "segmentation_method": segmentation_method,
        "mask_coverage_ratio": coverage,
        "modes": modes,
        "models": models,
        "image_models": IMAGE_MODELS,
        "image_pricing": IMAGE_PRICING,
        "analysis_pricing": ANALYSIS_PRICING,
        "output_image_tokens_4k": OUTPUT_IMAGE_TOKENS_4K,
        "planned_image_requests": request_count,
        "image_output_request_count": image_request_count,
        "image_output_only_cost_preview_usd": preview,
        "final_output_format": "PNG lossless",
        "status_schema": {
            "pipeline_status": "Overall requested pipeline outcome: success, success_unverified, partial_success, or failed.",
            "integration_status": "Final AI integration outcome: not_run, accepted, accepted_with_drift_warning, accepted_routing_unverified, accepted_with_drift_warning_routing_unverified, rejected_product_drift, or failed.",
            "final_source": "Source of the final_output_file: ai_integration, deterministic, or none.",
            "integration_rgb_delta": "Telemetry-only masked RGB difference between deterministic composite and final AI-integrated output; higher values can reflect beneficial relighting/blending.",
            "warning_message": "Non-fatal warnings such as RGB-delta drift telemetry or policy-based deterministic fallback.",
            "error_message": "Actual generation/integration errors.",
            "status": "Legacy detail status retained for older CSV/report consumers.",
        },
        "design_rule": (
            "No product-category hardcoding. "
            "Semantic analysis infers functional anchors and physical support; anchor-aware placement, geometry/perspective guards, deterministic color harmonization, and final AI integration prioritize real-world context and photorealism. Integration RGB Delta is telemetry/warning by default, not a visual-quality score or reject gate."
        ),
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    create_html_report(
        run_dir=run_dir,
        product_copy=product_copy,
        mask_path=mask_path,
        semantic=semantic,
        semantic_usage=semantic_usage,
        background_description=background_description,
        background_source=background_source,
        results_df=results_df,
        summary_df=summary_df,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Run folder     : {run_dir}")
    print(f"Results CSV    : {run_dir / 'results.csv'}")
    print(f"Summary CSV    : {run_dir / 'summary.csv'}")
    print(f"Manual review  : {run_dir / 'manual_review_template.csv'}")
    print(f"HTML report    : {run_dir / 'report.html'}")

    if not summary_df.empty:
        print()
        print("SUMMARY")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
