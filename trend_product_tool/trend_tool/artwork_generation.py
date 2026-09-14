from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .config import ProductTarget
from .product_asset import create_gemini_client, extract_image_bytes, image_part, is_transient_gemini_error


@dataclass(frozen=True)
class ArtworkGenerationRecord:
    source_path: Path
    output_path: Path | None
    model: str
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_flat_artwork(
    source: Path,
    destination: Path,
    target: ProductTarget,
    *,
    backend: str,
    model: str = "gemini-3-pro-image",
    image_size: str = "2K",
    attempts: int = 2,
    reference_brief: dict[str, object] | None = None,
    correction: str = "",
) -> ArtworkGenerationRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = create_gemini_client(backend)
        with Image.open(source) as opened:
            reference = opened.convert("RGB")
        from google.genai import types

        content = types.Content(
            role="user",
            parts=[image_part(reference, max_side=1536, max_bytes=3_500_000), types.Part.from_text(text=artwork_prompt(target, reference_brief, correction))],
        )
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            temperature=0.25,
            image_config=types.ImageConfig(
                aspect_ratio=artwork_aspect_ratio(target),
                image_size=image_size,
                output_mime_type="image/png",
            ),
        )
        last_error: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                response = client.models.generate_content(model=model, contents=[content], config=config)
                image_bytes, _ = extract_image_bytes(response)
                if not image_bytes:
                    raise RuntimeError("Gemini artwork generation returned no image.")
                with Image.open(io.BytesIO(image_bytes)) as generated:
                    generated.convert("RGBA").save(destination)
                return ArtworkGenerationRecord(source, destination, model, "ok", "Gemini generated flat printable artwork.")
            except Exception as exc:
                last_error = exc
                if attempt >= max(1, attempts) or not is_transient_gemini_error(exc):
                    break
                time.sleep(float(attempt) * 2.0)
        raise RuntimeError(str(last_error or "unknown Gemini error"))
    except Exception as exc:
        return ArtworkGenerationRecord(source, None, model, "failed", str(exc))


def artwork_prompt(target: ProductTarget, reference_brief: dict[str, object] | None = None, correction: str = "") -> str:
    product = target.name.strip().lower() or "rug"
    textile = "soft woven blanket textile" if product == "blanket" else "flat printed rug artwork"
    edge_rule = "Use a seamless repeat-safe composition with no border." if product == "blanket" else "Use a full-bleed composition that reaches all four edges, with no white margin."
    brief = json.dumps(reference_brief or {}, ensure_ascii=False)
    correction_rule = f"Previous attempt failed this quality check: {correction}\nCorrect that failure completely." if correction else ""
    return f"""
Create a new {textile} inspired by the attached reference image.
The reference is only a source of motif, palette, texture, and composition. Do not copy its product photography or scene.
Use only these extracted visual cues: {brief}
{edge_rule}
Requirements:
- output one newly created 2D graphic artwork image for printing;
- do not return, reproduce, or merely retouch the reference photograph;
- reinterpret the motifs as clean illustrated shapes, with a graphic-design or vector-art appearance;
- remove room, floor, wall, furniture, props, product edges, perspective, shadows, highlights, and camera distortion;
- preserve the useful visual language while creating a coherent original composition;
- fill the entire canvas with artwork;
- no mockup, no product photograph, no frame, no text, no logo, no watermark;
- keep details large enough for textile printing and use a vertical composition.
{correction_rule}
""".strip()


def artwork_aspect_ratio(target: ProductTarget) -> str:
    ratio = target.width_px / max(1, target.height_px)
    if ratio < 0.72:
        return "9:16"
    if ratio > 1.35:
        return "16:9"
    return "1:1"
