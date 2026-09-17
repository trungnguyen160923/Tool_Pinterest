from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


@dataclass(frozen=True)
class EnhancementRecord:
    source_path: Path
    output_path: Path
    mode: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    scale_factor: float
    source_sharpness: float
    output_sharpness: float
    latency_sec: float
    estimated_cost_usd: float
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def enhance_for_print(source: Path, destination: Path, min_long_edge: int, mode: str = "task2_local") -> EnhancementRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = (mode or "task2_local").strip().lower().replace("-", "_")
    started = time.perf_counter()
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
        source_size = image.size
        source_sharpness = sharpness_score(image)
        if mode in {"off", "none"}:
            output = image
            notes = "enhancement disabled"
        else:
            output = local_task2_style_enhance(image, min_long_edge)
            notes = "local faithful upscale/enhance for print production"
        output.save(destination)
    output_sharpness = sharpness_score(output)
    latency = time.perf_counter() - started
    return EnhancementRecord(
        source_path=source,
        output_path=destination,
        mode=mode,
        source_width=source_size[0],
        source_height=source_size[1],
        output_width=output.width,
        output_height=output.height,
        scale_factor=max(output.size) / max(1, max(source_size)),
        source_sharpness=round(source_sharpness, 4),
        output_sharpness=round(output_sharpness, 4),
        latency_sec=round(latency, 4),
        estimated_cost_usd=0.0,
        status="ok",
        notes=notes,
    )


def local_task2_style_enhance(image: Image.Image, min_long_edge: int) -> Image.Image:
    image = image.convert("RGBA")
    output = reduce_jpeg_artifacts(image)
    scale = max(1.0, min_long_edge / max(1, max(output.size)))
    if scale > 1:
        output = resize_in_steps(output, scale)
    rgb = output.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.03)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.035)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=3))
    rgb.putalpha(output.getchannel("A"))
    return rgb


def reduce_jpeg_artifacts(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    if max(image.size) < 900:
        rgb = rgb.filter(ImageFilter.MedianFilter(size=3))
    alpha = image.getchannel("A") if image.mode == "RGBA" else Image.new("L", image.size, 255)
    rgb.putalpha(alpha)
    return rgb


def resize_in_steps(image: Image.Image, scale: float) -> Image.Image:
    current = image
    remaining = scale
    while remaining > 2.01:
        current = current.resize((current.width * 2, current.height * 2), Image.Resampling.LANCZOS)
        remaining /= 2.0
    target_size = (
        max(1, round(current.width * remaining)),
        max(1, round(current.height * remaining)),
    )
    if target_size != current.size:
        current = current.resize(target_size, Image.Resampling.LANCZOS)
    return current


def sharpness_score(image: Image.Image) -> float:
    gray = image.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return float(stat.var[0])
