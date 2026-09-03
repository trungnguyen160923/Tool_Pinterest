from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageRecord:
    image_id: str
    path: str
    filename: str
    size_bytes: int
    mtime_ns: int
    width: int
    height: int
    aspect_ratio: float
    dhash: str
    color_hist: list[float]
    edge_hist: list[float]
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryFeatures:
    path: str
    width: int
    height: int
    aspect_ratio: float
    dhash: str
    color_hist: list[float]
    edge_hist: list[float]
    embedding: list[float] | None = None


@dataclass
class SimilarityResult:
    rank: int
    image_id: str
    path: str
    filename: str
    width: int
    height: int
    score: float
    semantic_score: float | None
    color_score: float
    edge_score: float
    hash_score: float
    aspect_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

