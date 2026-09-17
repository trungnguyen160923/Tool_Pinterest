from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter

from .crawler import CandidateImage
from .settings import env, load_tool_env


ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class ProductAssetConfig:
    output_dir: Path
    mode: str = "auto"
    gemini_backend: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    target_hint: str = "product"
    min_visible_percent: float = 80.0
    min_mask_coverage: float = 0.01
    max_mask_coverage: float = 0.70
    max_gemini_attempts: int = 3
    gemini_retry_delay_sec: float = 8.0
    reconstruct_rejected_assets: bool = True
    reconstruction_model: str = "gemini-3-pro-image"


@dataclass(frozen=True)
class ProductAsset:
    source_candidate: CandidateImage
    asset_path: Path
    mask_path: Path
    profile_path: Path
    profile: dict[str, Any]


@dataclass(frozen=True)
class ProductAssetRecord:
    source_path: Path
    asset_path: Path | None
    mask_path: Path | None
    profile_path: Path | None
    status: str
    reason: str
    profile: dict[str, Any]
    metrics: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationSelection:
    points: list[tuple[float, float]]
    source: str
    confidence: float
    label: str


def extract_product_assets(
    candidates: list[CandidateImage],
    config: ProductAssetConfig | None,
    progress: ProgressLogger | None = None,
) -> tuple[list[ProductAsset], list[ProductAssetRecord]]:
    if config is None or config.mode == "off":
        return [], []

    config.output_dir.mkdir(parents=True, exist_ok=True)
    client = None
    client_error = ""
    try:
        client = create_gemini_client(config.gemini_backend)
    except Exception as exc:
        client_error = str(exc)

    assets: list[ProductAsset] = []
    records: list[ProductAssetRecord] = []
    for index, candidate in enumerate(candidates, start=1):
        log(progress, f"[{index}/{len(candidates)}] Extracting clean product asset from {candidate.path.name}.")
        try:
            asset, record = extract_one_product_asset(candidate, config, client, client_error)
        except Exception as exc:
            record = write_rejection_record(candidate, config.output_dir, str(exc), {}, {})
            asset = None
        records.append(record)
        if asset is not None:
            assets.append(asset)
            log(progress, f"Accepted product asset: {asset.asset_path.name}.")
        else:
            log(progress, f"Rejected product asset: {candidate.path.name} ({record.reason}).")
    return assets, records


def extract_one_product_asset(
    candidate: CandidateImage,
    config: ProductAssetConfig,
    client: Any,
    client_error: str,
) -> tuple[ProductAsset | None, ProductAssetRecord]:
    with Image.open(candidate.path) as opened:
        original = opened.convert("RGBA")

    alpha = alpha_mask_from_image(original)
    profile: dict[str, Any] = {}
    method = "alpha_channel"
    selection: SegmentationSelection | None = None
    if alpha is None:
        if client is None:
            raise RuntimeError(f"Gemini product extraction unavailable: {client_error or 'client not configured'}")
        payload = call_gemini_product_analysis(
            client,
            config.gemini_model,
            original.convert("RGB"),
            config.target_hint,
            max_attempts=config.max_gemini_attempts,
            retry_delay_sec=config.gemini_retry_delay_sec,
        )
        profile = normalize_profile(payload.get("product_profile") or payload.get("profile") or {})
        selection = choose_product_polygon(payload, profile)
        alpha = mask_from_polygon(selection.points, original.size)
        method = f"gemini_{selection.source}"

    metrics = mask_metrics(alpha)
    if selection is not None:
        metrics.update(segmentation_metrics(selection))
    reason = validate_profile_and_mask(profile, metrics, config)
    stem = candidate.path.stem
    profile_path = config.output_dir / f"{stem}_profile.json"
    mask_path = config.output_dir / f"{stem}_mask.png"
    asset_path = config.output_dir / f"{stem}_product.png"

    cleanup_attempted = False
    if not reason and should_reconstruct_accepted_asset(profile, metrics, selection, config):
        cleanup_reason = accepted_asset_reconstruction_reason(profile, metrics, selection)
        cleanup_attempted = True
        if client is None:
            reason = f"{cleanup_reason}; AI cleanup unavailable: {client_error or 'client not configured'}"
        else:
            try:
                reconstructed = reconstruct_product_asset(
                    client=client,
                    model=config.reconstruction_model,
                    original=original.convert("RGB"),
                    profile=profile,
                    target_hint=config.target_hint,
                    reject_reason=cleanup_reason,
                    max_attempts=config.max_gemini_attempts,
                    retry_delay_sec=config.gemini_retry_delay_sec,
                )
                reconstructed_asset, reconstructed_mask = product_only_alpha_from_white_background(reconstructed)
                reconstructed_metrics = mask_metrics(reconstructed_mask)
                reconstructed_metrics.update(
                    {
                        "reconstructed_from_accepted_mask": True,
                        "accepted_cleanup_reason": cleanup_reason,
                    }
                )
                reconstructed_reason = validate_reconstructed_asset(reconstructed_metrics, config)
                if not reconstructed_reason:
                    asset_path = config.output_dir / f"{stem}_reconstructed_product.png"
                    mask_path = config.output_dir / f"{stem}_reconstructed_mask.png"
                    reconstructed_asset.save(asset_path)
                    reconstructed_mask.save(mask_path)
                    reconstructed_profile = dict(profile)
                    reconstructed_profile["asset_reconstruction"] = "gemini_image_model"
                    reconstructed_profile["accepted_cleanup_reason"] = cleanup_reason
                    write_json(
                        profile_path,
                        {
                            "profile": reconstructed_profile,
                            "metrics": reconstructed_metrics,
                            "status": "accepted",
                            "reason": "accepted_after_ai_cleanup",
                            "segmentation_method": "gemini_reconstructed_product",
                            "segmentation": segmentation_summary(selection),
                            "source_path": str(candidate.path),
                            "asset_path": str(asset_path),
                            "mask_path": str(mask_path),
                        },
                    )
                    asset = ProductAsset(candidate, asset_path, mask_path, profile_path, reconstructed_profile)
                    return asset, ProductAssetRecord(
                        source_path=candidate.path,
                        asset_path=asset_path,
                        mask_path=mask_path,
                        profile_path=profile_path,
                        status="accepted",
                        reason="accepted_after_ai_cleanup",
                        profile=reconstructed_profile,
                        metrics=reconstructed_metrics,
                    )
                metrics["accepted_reconstruction_reject_reason"] = reconstructed_reason
                reason = f"{cleanup_reason}; AI cleanup rejected: {reconstructed_reason}"
            except Exception as exc:
                metrics["accepted_reconstruction_error"] = str(exc)
                reason = f"{cleanup_reason}; AI cleanup failed: {exc}"

    if reason:
        if config.reconstruct_rejected_assets and client is not None and not cleanup_attempted:
            try:
                reconstructed = reconstruct_product_asset(
                    client=client,
                    model=config.reconstruction_model,
                    original=original.convert("RGB"),
                    profile=profile,
                    target_hint=config.target_hint,
                    reject_reason=reason,
                    max_attempts=config.max_gemini_attempts,
                    retry_delay_sec=config.gemini_retry_delay_sec,
                )
                reconstructed_asset, reconstructed_mask = product_only_alpha_from_white_background(reconstructed)
                reconstructed_metrics = mask_metrics(reconstructed_mask)
                reconstructed_metrics.update(
                    {
                        "reconstructed_from_rejected_mask": True,
                        "original_reject_reason": reason,
                    }
                )
                reconstructed_reason = validate_reconstructed_asset(reconstructed_metrics, config)
                if not reconstructed_reason:
                    asset_path = config.output_dir / f"{stem}_reconstructed_product.png"
                    mask_path = config.output_dir / f"{stem}_reconstructed_mask.png"
                    reconstructed_asset.save(asset_path)
                    reconstructed_mask.save(mask_path)
                    reconstructed_profile = dict(profile)
                    reconstructed_profile["asset_reconstruction"] = "gemini_image_model"
                    reconstructed_profile["original_reject_reason"] = reason
                    write_json(
                        profile_path,
                        {
                            "profile": reconstructed_profile,
                            "metrics": reconstructed_metrics,
                            "status": "accepted",
                            "reason": "accepted_after_ai_reconstruction",
                            "segmentation_method": "gemini_reconstructed_product",
                            "segmentation": segmentation_summary(selection),
                            "source_path": str(candidate.path),
                            "asset_path": str(asset_path),
                            "mask_path": str(mask_path),
                        },
                    )
                    asset = ProductAsset(candidate, asset_path, mask_path, profile_path, reconstructed_profile)
                    return asset, ProductAssetRecord(
                        source_path=candidate.path,
                        asset_path=asset_path,
                        mask_path=mask_path,
                        profile_path=profile_path,
                        status="accepted",
                        reason="accepted_after_ai_reconstruction",
                        profile=reconstructed_profile,
                        metrics=reconstructed_metrics,
                    )
                reason = f"{reason}; AI reconstruction rejected: {reconstructed_reason}"
            except Exception as exc:
                reason = f"{reason}; AI reconstruction failed: {exc}"

        review_asset_path = None
        review_mask_path = None
        if should_export_rejected_cutout(profile, metrics, config):
            review_dir = config.output_dir / "rejected_cutouts"
            review_mask_dir = config.output_dir / "rejected_masks"
            review_asset_path = review_dir / f"{stem}_product.png"
            review_mask_path = review_mask_dir / f"{stem}_mask.png"
            asset_img, cropped_mask = crop_asset(original, alpha)
            review_asset_path.parent.mkdir(parents=True, exist_ok=True)
            review_mask_path.parent.mkdir(parents=True, exist_ok=True)
            asset_img.save(review_asset_path)
            cropped_mask.save(review_mask_path)
        write_json(
            profile_path,
            {
                "profile": profile,
                "metrics": metrics,
                "status": "rejected",
                "reason": reason,
                "segmentation_method": method,
                "segmentation": segmentation_summary(selection),
                "source_path": str(candidate.path),
                "review_asset_path": str(review_asset_path) if review_asset_path else None,
                "review_mask_path": str(review_mask_path) if review_mask_path else None,
            },
        )
        return None, ProductAssetRecord(
            source_path=candidate.path,
            asset_path=review_asset_path,
            mask_path=review_mask_path,
            profile_path=profile_path,
            status="rejected",
            reason=reason,
            profile=profile,
            metrics=metrics,
        )

    asset_img, cropped_mask = crop_asset(original, alpha)
    asset_img.save(asset_path)
    cropped_mask.save(mask_path)
    write_json(
        profile_path,
        {
            "profile": profile,
            "metrics": metrics,
            "status": "accepted",
            "segmentation_method": method,
            "segmentation": segmentation_summary(selection),
            "source_path": str(candidate.path),
            "asset_path": str(asset_path),
            "mask_path": str(mask_path),
        },
    )
    asset = ProductAsset(candidate, asset_path, mask_path, profile_path, profile)
    record = ProductAssetRecord(
        source_path=candidate.path,
        asset_path=asset_path,
        mask_path=mask_path,
        profile_path=profile_path,
        status="accepted",
        reason="accepted",
        profile=profile,
        metrics=metrics,
    )
    return asset, record


def should_export_rejected_cutout(
    profile: dict[str, Any],
    metrics: dict[str, float | int | str],
    config: ProductAssetConfig,
) -> bool:
    coverage = safe_float(metrics.get("coverage"), 0.0)
    if coverage < config.min_mask_coverage or coverage > config.max_mask_coverage:
        return False
    if profile and bool(profile.get("mask_contains_non_product", False)):
        return False
    if int(metrics.get("border_touch_sides", 0)) >= 2:
        return False
    return True


def product_only_alpha_from_white_background(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    rgba = img.convert("RGBA")
    alpha = alpha_mask_from_image(rgba)
    if alpha is None:
        alpha = alpha_from_near_white_background(rgba.convert("RGB"))
        rgba.putalpha(alpha)
    asset, mask = crop_asset(rgba, alpha)
    return asset, mask


def alpha_from_near_white_background(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    alpha = Image.new("L", rgb.size, 0)
    pixels = rgb.load()
    alpha_pixels = alpha.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # Pure/near-white studio backgrounds become transparent; off-white product fibers remain mostly opaque.
            white_distance = max(255 - r, 255 - g, 255 - b)
            saturation = max(r, g, b) - min(r, g, b)
            if r >= 246 and g >= 246 and b >= 246 and saturation <= 8:
                value = 0
            elif r >= 238 and g >= 238 and b >= 238 and saturation <= 12:
                value = 80
            else:
                value = 255
            alpha_pixels[x, y] = value
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    return alpha


def validate_reconstructed_asset(metrics: dict[str, float | int | str], config: ProductAssetConfig) -> str:
    coverage = safe_float(metrics.get("coverage"), 0.0)
    if coverage < config.min_mask_coverage:
        return f"reconstructed mask coverage {coverage:.3f} is too small."
    if coverage > 0.92:
        return f"reconstructed mask coverage {coverage:.3f} is too large."
    if int(metrics.get("border_touch_sides", 0)) >= 3:
        return "reconstructed product touches too many image borders."
    return ""


def create_gemini_client(backend: str) -> Any:
    load_tool_env()
    from google import genai
    from google.genai import types

    backend = backend.strip().lower()
    api_key = env("GEMINI_API_KEY", "").strip() or env("GOOGLE_API_KEY", "").strip()
    if backend == "api-key" or (backend == "auto" and api_key):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for api-key backend.")
        return genai.Client(api_key=api_key)

    project = env("GOOGLE_CLOUD_PROJECT", "").strip()
    location = env("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for enterprise Gemini backend.")
    return genai.Client(
        enterprise=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1", timeout=600_000),
    )


def call_gemini_product_analysis(
    client: Any,
    model: str,
    image: Image.Image,
    target_hint: str,
    *,
    max_attempts: int = 3,
    retry_delay_sec: float = 8.0,
) -> dict[str, Any]:
    from google.genai import types

    prompt = product_analysis_prompt(target_hint)
    part = image_part(image)
    start = time.perf_counter()
    attempts = max(1, int(max_attempts))
    response = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[part, types.Part.from_text(text=prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.05,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            if attempt >= attempts or not is_transient_gemini_error(exc):
                raise
            time.sleep(max(1.0, retry_delay_sec) * attempt)
    if response is None:
        raise RuntimeError("Gemini product analysis did not return a response.")
    raw_text = extract_response_text(response)
    try:
        payload = parse_json_relaxed(raw_text)
    except Exception as exc:
        try:
            payload = parse_product_analysis_salvage(raw_text)
        except Exception:
            payload = None
        if payload is not None:
            payload["_latency_sec"] = round(time.perf_counter() - start, 4)
            payload["_parse_recovery"] = "salvaged_profile_and_box"
            return payload
        preview = " ".join(raw_text.split())[:700]
        raise ValueError(f"Invalid Gemini product-analysis JSON: {exc}; raw preview: {preview}") from exc
    payload["_latency_sec"] = round(time.perf_counter() - start, 4)
    return payload


def product_analysis_prompt(target_hint: str) -> str:
    hint = target_hint.strip() or "product"
    return f"""
Analyze this ecommerce/lifestyle image and isolate the main sellable product.

Target hint from the workflow: {hint}

Do not force the hint if the image clearly shows a different sellable product.
Return JSON only with this shape:
{{
  "product_profile": {{
    "product_label": "short label",
    "physical_form": "PLANAR_FLEXIBLE | PLANAR_RIGID | RIGID_OBJECT | APPAREL | SOFT_GOODS | UNKNOWN",
    "natural_support": "FLOOR | WALL | TABLE | BODY | HAND | NONE | UNKNOWN",
    "canonical_view": "TOP_DOWN | OBLIQUE_TOP_DOWN | FRONT | SIDE | THREE_QUARTER | UNKNOWN",
    "placement_strategy": "PLANAR_SURFACE | VERTICAL_SURFACE | OBJECT_ON_SURFACE | WORN | UNKNOWN",
    "shape_type": "RECTANGLE | IRREGULAR_CONTOUR | COMPACT_OBJECT | UNKNOWN",
    "segmentation_target": "main sellable product only",
    "forbidden_mask_objects": ["support surfaces, props, furniture, people, packaging, text, logos"],
    "occlusion_level": "NONE | LOW | MEDIUM | HIGH",
    "visible_percent": 0,
    "crop_status": "NONE | MINOR_EDGE_CROP | MAJOR_CROP",
    "mask_contains_non_product": false,
    "should_accept": true,
    "reject_reason": null
  }},
  "segments": [
    {{
      "label": "main sellable product",
      "confidence": 0.0,
      "box_2d": [0, 0, 1000, 1000],
      "polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    }}
  ]
}}

Segmentation rules:
- The polygon or box_2d must include only the main sellable product.
- Return only short JSON. Do not return bitmap masks, RLE strings, compressed masks, base64, SVG, or long encoded data.
- If the product is approximately rectangular, return a 4-point polygon and matching box_2d.
- Exclude furniture, room background, support surface, floor, wall, table, sofa, chair, plants, hands, props, shadows, text, logos, and packaging.
- For lifestyle images where the product is draped, folded, worn, held, or sitting on props/furniture, the polygon must trace the visible product boundary precisely with enough points. Do not return a coarse triangle/quadrilateral or a bounding region that includes background/props.
- If the product boundary is unclear, heavily occluded, cropped, or mixed with non-product objects, set should_accept=false and explain reject_reason.
- Prefer rejecting a bad product asset over returning a polygon that includes non-product pixels.
""".strip()


def reconstruct_product_asset(
    *,
    client: Any,
    model: str,
    original: Image.Image,
    profile: dict[str, Any],
    target_hint: str,
    reject_reason: str,
    max_attempts: int,
    retry_delay_sec: float,
) -> Image.Image:
    from google.genai import types

    prompt = product_reconstruction_prompt(target_hint, profile, reject_reason)
    content = types.Content(
        role="user",
        parts=[
            image_part(original, max_side=1536, max_bytes=3_500_000),
            types.Part.from_text(text=prompt),
        ],
    )
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.08,
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="1K",
            output_mime_type="image/png",
        ),
    )
    last_error: Exception | None = None
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[content],
                config=config,
            )
            image_bytes, _ = extract_image_bytes(response)
            if not image_bytes:
                raise RuntimeError("Gemini reconstruction returned no image.")
            with Image.open(io.BytesIO(image_bytes)) as opened:
                return opened.convert("RGBA")
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not is_transient_gemini_error(exc):
                break
            time.sleep(max(1.0, retry_delay_sec) * attempt)
    raise RuntimeError(f"Gemini product reconstruction failed after {attempts} attempt(s): {last_error}")


def product_reconstruction_prompt(target_hint: str, profile: dict[str, Any], reject_reason: str) -> str:
    hint = target_hint.strip() or str(profile.get("product_label") or "product").strip() or "product"
    label = str(profile.get("product_label") or hint).strip()
    physical_form = str(profile.get("physical_form") or "UNKNOWN").strip()
    shape_type = str(profile.get("shape_type") or "UNKNOWN").strip()
    return f"""
Create a clean ecommerce product-only image from the reference.

Target product: {hint}
Detected product label: {label}
Physical form: {physical_form}
Shape type: {shape_type}
Reason the original mask failed: {reject_reason}

Requirements:
- Output only the main sellable product, not the room, floor, furniture, props, hands, packaging, text, logos, watermark, or background.
- Preserve the product's visible design, colors, texture, material, proportions, and distinctive pattern as faithfully as possible.
- Preserve every motif or decorative element that is printed, tufted, stitched, appliqued, raised from, or visually integrated with the product surface. For rugs, do not remove central characters, seasonal icons, optical-illusion elements, borders, or featured artwork that make the product recognizable.
- If the reference is a lifestyle image, reconstruct the complete product in a normal catalog view suitable for resale.
- Center the whole product with comfortable margins. Do not crop any edge.
- Use a pure white seamless background or transparent background if supported.
- Do not add decorative scene elements, room props, shadows that look like furniture/floor, labels, text, or extra products. If an object appears attached to or intentionally part of the product design, preserve it as product artwork instead of treating it as a scene prop.
- Return one product image only.
""".strip()


def extract_image_bytes(response: Any) -> tuple[bytes | None, str]:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or "image/png"
            if data:
                if isinstance(data, str):
                    return base64.b64decode(data), mime
                return bytes(data), mime
    payload = obj_to_dict(response)
    return extract_image_bytes_from_obj(payload)


def extract_image_bytes_from_obj(obj: Any) -> tuple[bytes | None, str]:
    if isinstance(obj, dict):
        inline = obj.get("inlineData") or obj.get("inline_data")
        if isinstance(inline, dict):
            data = inline.get("data")
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            if data:
                if isinstance(data, str):
                    try:
                        return base64.b64decode(data), mime
                    except Exception:
                        pass
                elif isinstance(data, (bytes, bytearray)):
                    return bytes(data), mime
        for value in obj.values():
            found = extract_image_bytes_from_obj(value)
            if found[0] is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = extract_image_bytes_from_obj(item)
            if found[0] is not None:
                return found
    return None, ""


def obj_to_dict(obj: Any) -> dict[str, Any]:
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


def image_part(img: Image.Image, max_side: int = 1536, max_bytes: int = 3_500_000) -> Any:
    from google.genai import types

    working = img.convert("RGB")
    w, h = working.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        working = working.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    quality = 88
    data = b""
    while quality >= 50:
        buf = io.BytesIO()
        working.save(buf, "JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            break
        quality -= 8
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "\n".join(chunks)


def parse_json_relaxed(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    candidates = [stripped]
    balanced = extract_balanced_json_object(stripped)
    if balanced and balanced != stripped:
        candidates.append(balanced)
    last_error: Exception | None = None
    for candidate in candidates:
        for variant in unique_text_candidates([candidate, repair_json_text(candidate)]):
            try:
                data = json.loads(variant)
                if not isinstance(data, dict):
                    raise ValueError("Gemini product analysis did not return a JSON object.")
                return data
            except Exception as exc:
                last_error = exc
    if last_error:
        raise last_error
    raise ValueError("Gemini product analysis did not return JSON text.")


def parse_product_analysis_salvage(text: str) -> dict[str, Any]:
    profile_match = re.search(
        r'"product_profile"\s*:\s*(\{.*?\})\s*,\s*"segments"',
        text,
        flags=re.DOTALL,
    )
    box_match = re.search(r'"box_2d"\s*:\s*\[([^\]]+)\]', text, flags=re.DOTALL)
    if not profile_match or not box_match:
        raise ValueError("No salvageable product profile and box_2d found.")
    profile = json.loads(repair_json_text(profile_match.group(1)))
    box_values = [float(item.strip()) for item in box_match.group(1).split(",")[:4]]
    if len(box_values) != 4:
        raise ValueError("Salvaged box_2d does not have four values.")
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    segment = {
        "label": label_match.group(1) if label_match else "main sellable product",
        "confidence": safe_float(confidence_match.group(1) if confidence_match else None, 0.5),
        "box_2d": box_values,
    }
    return {"product_profile": profile, "segments": [segment]}


def extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def repair_json_text(text: str) -> str:
    repaired = text.replace("\ufeff", "").strip()
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r'((?:true|false|null)|[}\]"0-9])\s*\n\s*(?="[^"\n]+"\s*:)',
        r"\1,\n",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r'([}\]"0-9])\s*\n\s*(?=[{\[])',
        r"\1,\n",
        repaired,
    )
    return repaired


def unique_text_candidates(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def normalize_profile(raw: dict[str, Any]) -> dict[str, Any]:
    profile = dict(raw)
    profile.setdefault("should_accept", True)
    profile.setdefault("visible_percent", 0)
    profile.setdefault("occlusion_level", "UNKNOWN")
    profile.setdefault("crop_status", "UNKNOWN")
    profile.setdefault("mask_contains_non_product", False)
    profile.setdefault("reject_reason", None)
    return profile


def choose_product_polygon(payload: dict[str, Any], profile: dict[str, Any]) -> SegmentationSelection:
    segments = payload.get("segments") or payload.get("masks") or payload.get("boxes") or []
    if not isinstance(segments, list):
        raise ValueError("Gemini segmentation response has no segment list.")
    forbidden = [str(item).lower() for item in profile.get("forbidden_mask_objects", []) if str(item).strip()]
    candidates: list[tuple[float, SegmentationSelection]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").lower()
        if forbidden and any(word and word in label for word in forbidden):
            continue
        polygon = item.get("mask") or item.get("polygon") or item.get("mask_polygon")
        points = normalize_polygon(polygon)
        source = "polygon" if len(points) >= 3 else ""
        if len(points) < 3:
            points = polygon_from_box(item.get("box_2d") or item.get("bbox_2d") or item.get("box"))
            source = "box" if len(points) >= 3 else ""
            if points and str(profile.get("shape_type", "")).upper() == "IRREGULAR_CONTOUR":
                continue
        if len(points) < 3:
            continue
        confidence = safe_float(item.get("confidence"), 0.5)
        score = confidence * max(1.0, polygon_area(points))
        candidates.append(
            (
                score,
                SegmentationSelection(
                    points=points,
                    source=source,
                    confidence=confidence,
                    label=label or "main sellable product",
                ),
            )
        )
    if not candidates:
        raise ValueError("Gemini did not return a usable product polygon.")
    return max(candidates, key=lambda item: item[0])[1]


def polygon_from_box(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        y1, x1, y2, x2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return []
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def normalize_polygon(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return points


def mask_from_polygon(points: list[tuple[float, float]], size: tuple[int, int]) -> Image.Image:
    width, height = size
    scaled = [
        (round(max(0.0, min(1000.0, x)) / 1000.0 * (width - 1)), round(max(0.0, min(1000.0, y)) / 1000.0 * (height - 1)))
        for x, y in points
    ]
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(scaled, fill=255)
    return mask


def alpha_mask_from_image(img: Image.Image) -> Image.Image | None:
    if img.mode != "RGBA":
        return None
    alpha = img.getchannel("A")
    if alpha.getextrema()[0] >= 250:
        return None
    return alpha


def mask_metrics(mask: Image.Image) -> dict[str, float | int | str]:
    mask = mask.convert("L")
    width, height = mask.size
    bbox = mask.getbbox()
    if bbox is None:
        return {"coverage": 0.0, "bbox_area": 0.0, "border_touch_sides": 0}
    active = 0
    for value in mask.getdata():
        if value > 16:
            active += 1
    coverage = active / float(width * height)
    left, top, right, bottom = bbox
    border_touch_sides = int(left <= 1) + int(top <= 1) + int(right >= width - 1) + int(bottom >= height - 1)
    bbox_area = ((right - left) * (bottom - top)) / float(width * height)
    return {
        "coverage": round(coverage, 6),
        "bbox_area": round(bbox_area, 6),
        "border_touch_sides": border_touch_sides,
        "bbox_left": left,
        "bbox_top": top,
        "bbox_right": right,
        "bbox_bottom": bottom,
    }


def segmentation_metrics(selection: SegmentationSelection) -> dict[str, float | int | str]:
    return {
        "segmentation_source": selection.source,
        "polygon_points": len(selection.points),
        "polygon_bbox_fill_ratio": round(polygon_bbox_fill_ratio(selection.points), 6),
        "segment_confidence": round(selection.confidence, 4),
        "segment_label": selection.label,
    }


def segmentation_summary(selection: SegmentationSelection | None) -> dict[str, float | int | str] | None:
    if selection is None:
        return None
    return {
        "source": selection.source,
        "points": len(selection.points),
        "bbox_fill_ratio": round(polygon_bbox_fill_ratio(selection.points), 6),
        "confidence": round(selection.confidence, 4),
        "label": selection.label,
    }


def validate_profile_and_mask(
    profile: dict[str, Any],
    metrics: dict[str, float | int | str],
    config: ProductAssetConfig,
) -> str:
    coverage = safe_float(metrics.get("coverage"), 0.0)
    visible = safe_float(profile.get("visible_percent"), 100.0 if not profile else 0.0)
    occlusion = str(profile.get("occlusion_level", "UNKNOWN")).upper()
    crop_status = str(profile.get("crop_status", "UNKNOWN")).upper()
    if profile and not bool(profile.get("should_accept", True)):
        return str(profile.get("reject_reason") or "Gemini rejected image for product extraction.")
    if profile and bool(profile.get("mask_contains_non_product", False)):
        return "Gemini detected non-product objects inside the product mask."
    if profile and visible < config.min_visible_percent:
        return f"visible_percent {visible:.1f} is below {config.min_visible_percent:.1f}."
    if occlusion == "HIGH":
        return "product occlusion is HIGH."
    if crop_status == "MAJOR_CROP":
        return "product has major crop."
    if coverage < config.min_mask_coverage:
        return f"mask coverage {coverage:.3f} is too small."
    if coverage > config.max_mask_coverage:
        return f"mask coverage {coverage:.3f} is too large."
    if int(metrics.get("border_touch_sides", 0)) >= 2:
        return "mask touches multiple image borders, likely cropped or includes background."
    source = str(metrics.get("segmentation_source", "")).lower()
    polygon_points = int(safe_float(metrics.get("polygon_points"), 0.0))
    fill_ratio = safe_float(metrics.get("polygon_bbox_fill_ratio"), 1.0)
    if source == "box" and not is_flat_planar_profile(profile):
        return "Gemini returned only a bounding box for a non-flat/lifestyle product; cutout would include background."
    if source == "polygon" and polygon_points <= 4 and fill_ratio < 0.72 and not is_flat_planar_profile(profile):
        return "Gemini returned a coarse polygon for a non-flat/lifestyle product; cutout would include background."
    return ""


def should_reconstruct_accepted_asset(
    profile: dict[str, Any],
    metrics: dict[str, float | int | str],
    selection: SegmentationSelection | None,
    config: ProductAssetConfig,
) -> bool:
    if not config.reconstruct_rejected_assets or selection is None:
        return False
    if not is_flat_planar_profile(profile):
        return False
    canonical_view = str(profile.get("canonical_view", "UNKNOWN")).upper()
    source = str(metrics.get("segmentation_source", "")).lower()
    polygon_points = int(safe_float(metrics.get("polygon_points"), 0.0))
    fill_ratio = safe_float(metrics.get("polygon_bbox_fill_ratio"), 1.0)
    border_touch_sides = int(metrics.get("border_touch_sides", 0))
    if canonical_view != "TOP_DOWN":
        return True
    if source == "box":
        return True
    if source == "polygon" and polygon_points <= 4 and fill_ratio < 0.95:
        return True
    if border_touch_sides > 0:
        return True
    return False


def accepted_asset_reconstruction_reason(
    profile: dict[str, Any],
    metrics: dict[str, float | int | str],
    selection: SegmentationSelection | None,
) -> str:
    canonical_view = str(profile.get("canonical_view", "UNKNOWN")).upper()
    source = str(metrics.get("segmentation_source", "unknown"))
    polygon_points = int(safe_float(metrics.get("polygon_points"), 0.0))
    fill_ratio = safe_float(metrics.get("polygon_bbox_fill_ratio"), 1.0)
    border_touch_sides = int(metrics.get("border_touch_sides", 0))
    label = selection.label if selection is not None else "main sellable product"
    return (
        "Accepted segmentation still needs catalog cleanup: "
        f"label={label}, view={canonical_view}, segmentation={source}, "
        f"points={polygon_points}, bbox_fill_ratio={fill_ratio:.3f}, "
        f"border_touch_sides={border_touch_sides}. "
        "Remove lifestyle background and reconstruct a clean complete product-only asset."
    )


def is_flat_planar_profile(profile: dict[str, Any]) -> bool:
    if not profile:
        return False
    physical_form = str(profile.get("physical_form", "")).upper()
    placement = str(profile.get("placement_strategy", "")).upper()
    canonical_view = str(profile.get("canonical_view", "")).upper()
    natural_support = str(profile.get("natural_support", "")).upper()
    if physical_form not in {"PLANAR_FLEXIBLE", "PLANAR_RIGID"}:
        return False
    if placement not in {"PLANAR_SURFACE", "VERTICAL_SURFACE"}:
        return False
    if canonical_view not in {"TOP_DOWN", "OBLIQUE_TOP_DOWN", "FRONT", "SIDE"}:
        return False
    return natural_support in {"FLOOR", "WALL", "TABLE", "NONE", "UNKNOWN"}


def is_transient_gemini_error(exc: Exception) -> bool:
    message = str(exc).upper()
    transient_markers = [
        "429",
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "503",
        "504",
    ]
    return any(marker in message for marker in transient_markers)


def crop_asset(img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    mask = mask.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Cannot crop empty product mask.")
    width, height = img.size
    pad = max(2, round(min(width, height) * 0.015))
    left, top, right, bottom = bbox
    crop_box = (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad))
    cropped = img.crop(crop_box)
    cropped_mask = mask.crop(crop_box)
    cropped.putalpha(cropped_mask)
    return cropped, cropped_mask


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bbox_fill_ratio(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox_area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
    if bbox_area <= 0:
        return 0.0
    return polygon_area(points) / bbox_area


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_rejection_record(
    candidate: CandidateImage,
    output_dir: Path,
    reason: str,
    profile: dict[str, Any],
    metrics: dict[str, float | int | str],
) -> ProductAssetRecord:
    profile_path = output_dir / f"{candidate.path.stem}_profile.json"
    write_json(
        profile_path,
        {
            "profile": profile,
            "metrics": metrics,
            "status": "rejected",
            "reason": reason,
            "source_path": str(candidate.path),
        },
    )
    return ProductAssetRecord(
        source_path=candidate.path,
        asset_path=None,
        mask_path=None,
        profile_path=profile_path,
        status="rejected",
        reason=reason,
        profile=profile,
        metrics=metrics,
    )


def log(progress: ProgressLogger | None, message: str) -> None:
    if progress:
        progress(message)
