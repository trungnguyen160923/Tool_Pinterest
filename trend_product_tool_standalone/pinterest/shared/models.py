from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuerySpec:
    query: str
    intent: str = "product"
    priority: int = 1


@dataclass
class TrendPackageItem:
    trend_id: str
    trend: str
    trend_strength: float
    relationship: str
    semantic_fit: float
    queries: list[QuerySpec]
    reason: str = ""
    sources: list[str] = field(default_factory=list)
    source_metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class TrendPackage:
    schema_version: str
    generated_at: str
    niche: str
    region: str
    source: dict[str, Any]
    trends: list[TrendPackageItem]
    rejected_trends: list[dict[str, Any]] = field(default_factory=list)
    raw_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    result_id: str
    query: str
    trend_id: str
    trend: str
    image_url: str
    pin_url: str = ""
    pin_id: str = ""
    title: str = ""
    description: str = ""
    source: str = ""
    width: int | None = None
    height: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageCandidate:
    image_id: str
    query: str
    trend_id: str
    trend: str
    image_url: str
    pin_url: str = ""
    pin_id: str = ""
    title: str = ""
    source: str = ""
    local_path: str = ""
    width: int | None = None
    height: int | None = None
    dhash: str = ""
    download_error: str = ""
    duplicate_of: str = ""
    trend_strength: float = 0.0
    semantic_fit: float = 0.0


@dataclass
class VisionResult:
    image_id: str
    accepted: bool
    product_present: bool
    product_role: str
    product_confidence: float
    product_visibility: float
    trend_relevance: float
    commercial_quality: float
    aesthetic: str
    detected_product: str
    reason: str
    confidence: float
    main_subject: str = ""
    target_product_type: str = ""
    is_single_product: bool = False
    is_physical_product: bool = False
    is_floor_textile: bool = False
    is_collage: bool = False
    is_doormat: bool = False
    is_bath_mat: bool = False
    is_wall_tapestry: bool = False
    motifs: list[str] = field(default_factory=list)
    reject_reason_code: str = ""
    policy_reject_reason: str = ""
    error: str = ""
    source_role: str = "unknown"
    is_lifestyle_scene: bool = False
    foreground_coverage: float = 0.0
    background_complexity: float = 0.0
    flat_artwork_score: float = 0.0
    printability_score: float = 0.0
    requires_extraction: bool = False


@dataclass
class RankedImage:
    rank: int
    image_id: str
    image_score: float
    trend_id: str
    trend: str
    query: str
    image_url: str
    local_path: str
    pin_url: str
    pin_id: str
    width: int | None
    height: int | None
    product_role: str
    product_confidence: float
    product_visibility: float
    trend_relevance: float
    commercial_quality: float
    trend_strength: float
    semantic_fit: float
    source: str
    aesthetic: str
    detected_product: str
    reason: str
    main_subject: str = ""
    target_product_type: str = ""
    motifs: list[str] = field(default_factory=list)
    source_role: str = "unknown"
    is_lifestyle_scene: bool = False
    foreground_coverage: float = 0.0
    background_complexity: float = 0.0
    flat_artwork_score: float = 0.0
    printability_score: float = 0.0
    requires_extraction: bool = False
    classification: str = "Printable Artwork"
    is_direct_printable: bool = False

