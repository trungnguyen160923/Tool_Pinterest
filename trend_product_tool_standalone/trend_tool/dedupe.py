from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from .crawler import CandidateImage


@dataclass(frozen=True)
class DedupeDecision:
    candidate: CandidateImage
    kept: bool
    reason: str
    distance: int | None = None
    duplicate_of: Path | None = None


def dedupe_candidates(candidates: list[CandidateImage], threshold: int) -> tuple[list[CandidateImage], list[DedupeDecision]]:
    hashes: list[tuple[CandidateImage, int]] = []
    decisions: list[DedupeDecision] = []
    kept: list[CandidateImage] = []

    for candidate in candidates:
        try:
            image_hash = average_hash(candidate.path)
        except Exception as exc:
            decisions.append(DedupeDecision(candidate, False, f"unreadable: {exc}"))
            continue

        duplicate = None
        best_distance = None
        for kept_candidate, kept_hash in hashes:
            distance = hamming_distance(image_hash, kept_hash)
            if best_distance is None or distance < best_distance:
                best_distance = distance
            if distance <= threshold:
                duplicate = kept_candidate
                break

        if duplicate:
            decisions.append(DedupeDecision(candidate, False, "near_duplicate", best_distance, duplicate.path))
            continue

        hashes.append((candidate, image_hash))
        kept.append(candidate)
        decisions.append(DedupeDecision(candidate, True, "kept", best_distance))

    return kept, decisions


def average_hash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_low_information(path: Path, min_stddev: float = 10.0) -> bool:
    with Image.open(path) as image:
        stat = ImageStat.Stat(image.convert("L").resize((128, 128)))
    return bool(stat.stddev[0] < min_stddev)

