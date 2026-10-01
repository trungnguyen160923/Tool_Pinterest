from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import ProductTarget
from .product_asset import create_gemini_client, extract_response_text, image_part, parse_json_relaxed


AUTO_RENDERABLE_SHAPES = {"rectangle", "square", "round", "oval", "runner", "arch", "organic", "custom_cutline"}


@dataclass(frozen=True)
class RugShapeDecision:
    shape: str
    confidence: float
    reason: str
    alternatives: tuple[str, ...]
    custom_cutline_recommended: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def recommend_rug_shape(
    artwork_path: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str,
) -> RugShapeDecision:
    if target.name.strip().lower() != "rug":
        return RugShapeDecision("rectangle", 100.0, "Shape analysis applies only to rugs.", ())
    try:
        with Image.open(artwork_path) as opened:
            artwork = ImageOps.exif_transpose(opened).convert("RGB")
        assessment = _assess(artwork, backend=backend, model=model)
        recommended = str(assessment.get("recommended_shape") or "rectangle").strip().lower()
        custom = recommended in {"custom_cutline", "organic"} or bool(assessment.get("custom_cutline_recommended"))
        shape = recommended if recommended in AUTO_RENDERABLE_SHAPES else "rectangle"
        alternatives = tuple(
            value for value in (str(item).strip().lower() for item in assessment.get("alternatives", []))
            if value in AUTO_RENDERABLE_SHAPES and value != shape
        )
        confidence = max(0.0, min(100.0, float(assessment.get("confidence") or 0)))
        reason = str(assessment.get("reason") or "AI selected the safest standard rug silhouette.")
        if custom and shape not in {"organic", "custom_cutline"}:
            reason += " Custom cutline was suggested but requires a vendor-ready contour, so rectangle is used automatically."
        return RugShapeDecision(shape, confidence, reason, alternatives, custom)
    except Exception as exc:
        return RugShapeDecision("rectangle", 0.0, f"Shape analysis unavailable; using rectangle: {exc}", ())


def _assess(artwork: Image.Image, *, backend: str, model: str) -> dict[str, Any]:
    from google.genai import types

    client = create_gemini_client(backend)
    response = client.models.generate_content(
        model=model,
        contents=[
            image_part(artwork),
            """Analyze this flat rug artwork's composition for the most commercially suitable physical rug silhouette.
Choose one recommended_shape: rectangle, square, round, oval, runner, arch, or organic (custom_cutline).
Use round for radial/central motifs; square for balanced all-over or central compositions; oval for soft horizontal compositions; runner for strongly vertical artwork; arch for arched top/doorway compositions; organic for irregular/freeform curved compositions; rectangle for ordinary directional or full-bleed designs.
Return JSON only:
{
  "recommended_shape": "rectangle",
  "confidence": 0,
  "alternatives": ["square"],
  "custom_cutline_recommended": false,
  "reason": "short visual rationale"
}""",
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    parsed = parse_json_relaxed(extract_response_text(response))
    return parsed if isinstance(parsed, dict) else {}
