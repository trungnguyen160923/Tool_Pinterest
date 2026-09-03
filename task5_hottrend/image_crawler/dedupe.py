from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from ..shared.models import ImageCandidate


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
        return 999


def dedupe_candidates(candidates: list[ImageCandidate], *, dhash_distance: int = 5) -> tuple[list[ImageCandidate], list[ImageCandidate]]:
    kept: list[ImageCandidate] = []
    rejected: list[ImageCandidate] = []
    for candidate in candidates:
        if not candidate.local_path or candidate.download_error:
            rejected.append(candidate)
            continue
        try:
            with Image.open(Path(candidate.local_path)) as image:
                candidate.dhash = dhash_hex(image)
                candidate.width, candidate.height = image.size
        except Exception as exc:
            candidate.download_error = str(exc)
            rejected.append(candidate)
            continue
        duplicate = None
        for existing in kept:
            if existing.dhash and hamming_hex(existing.dhash, candidate.dhash) <= dhash_distance:
                duplicate = existing
                break
        if duplicate:
            candidate.duplicate_of = duplicate.image_id
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected

