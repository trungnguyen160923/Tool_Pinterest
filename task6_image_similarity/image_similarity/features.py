from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps

from .models import QueryFeatures


def load_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")


def dhash_hex(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:0{math.ceil(hash_size * hash_size / 4)}x}"


def hamming_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 64


def color_histogram(image: Image.Image, bins: int = 8, size: int = 96) -> list[float]:
    small = image.resize((size, size), Image.Resampling.BILINEAR)
    counts = [0] * (bins * bins * bins)
    step = 256 / bins
    for red, green, blue in small.getdata():
        r_bin = min(bins - 1, int(red / step))
        g_bin = min(bins - 1, int(green / step))
        b_bin = min(bins - 1, int(blue / step))
        counts[(r_bin * bins + g_bin) * bins + b_bin] += 1
    total = float(size * size) or 1.0
    return [count / total for count in counts]


def edge_histogram(image: Image.Image, bins: int = 8, size: int = 96) -> list[float]:
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    counts = [0.0] * bins

    for y in range(1, size - 1):
        row = y * size
        prev_row = (y - 1) * size
        next_row = (y + 1) * size
        for x in range(1, size - 1):
            gx = pixels[row + x + 1] - pixels[row + x - 1]
            gy = pixels[next_row + x] - pixels[prev_row + x]
            magnitude = math.hypot(gx, gy)
            if magnitude < 8:
                continue
            angle = (math.atan2(gy, gx) + math.pi) / (2 * math.pi)
            bucket = min(bins - 1, int(angle * bins))
            counts[bucket] += magnitude

    total = sum(counts) or 1.0
    return [count / total for count in counts]


def extract_query_features(path: Path) -> QueryFeatures:
    path = path.resolve()
    with load_rgb_image(path) as image:
        width, height = image.size
        aspect_ratio = width / height if height else 1.0
        return QueryFeatures(
            path=str(path),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            dhash=dhash_hex(image),
            color_hist=color_histogram(image),
            edge_hist=edge_histogram(image),
        )


def histogram_intersection(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return max(0.0, min(1.0, sum(min(x, y) for x, y in zip(a, b))))


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return None
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def hash_similarity(a: str, b: str, bits: int = 64) -> float:
    return max(0.0, min(1.0, 1.0 - hamming_hex(a, b) / bits))


def aspect_similarity(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    # Same ratio is 1.0; a 4x ratio mismatch trends toward 0.0.
    return max(0.0, min(1.0, 1.0 - abs(math.log(a / b)) / math.log(4.0)))
