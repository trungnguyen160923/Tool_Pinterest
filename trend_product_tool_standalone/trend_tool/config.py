from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductTarget:
    name: str
    width_px: int
    height_px: int
    dpi: int = 300
    prefer_cmyk: bool = True
    allow_custom_shape: bool = False
    rug_shape: str = "rectangle"


@dataclass(frozen=True)
class CrawlQuery:
    keyword: str
    max_results: int = 24


@dataclass(frozen=True)
class PipelineConfig:
    target: ProductTarget
    output_root: Path
    workflow_mode: str = "trend_to_product"
    trend_niche: str = ""
    trend_region: str = "US"
    trend_type: str = "growing"
    trend_interest: str = ""
    trend_keyword_limit: int = 50
    trend_max_trends: int = 20
    trend_min_semantic_fit: float = 35.0
    trend_max_queries_per_trend: int = 6
    desired_output_count: int = 5
    gemini_backend: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    task5_token_path: Path | None = None
    task5_provider: str = "pinterest-browser"
    task5_max_images_per_query: int = 30
    task5_max_crawl_trends: int = 5
    task5_max_downloads: int = 250
    task5_top_images: int = 100
    task5_vision_mode: str = "auto"
    task5_crawl_purpose: str = "auto"
    task5_product_focus: str = "auto"
    task5_refresh_vision_cache: bool = True
    network_preflight: bool = True
    task6_gallery: Path | None = None
    task6_index_path: Path | None = None
    task6_metadata_path: Path | None = None
    task6_similarity_threshold: float = 92.0
    task6_rebuild_index: bool = False
    task6_mode: str = "auto"
    task6_clip_model: str = "openai/clip-vit-base-patch32"
    task6_device: str = "auto"
    product_asset_mode: str = "auto"
    product_asset_min_visible_percent: float = 80.0
    product_asset_min_mask_coverage: float = 0.01
    product_asset_max_mask_coverage: float = 0.70
    design_mode: str = "ai_artwork"
    artwork_image_size: str = "2K"
    enhancement_mode: str = "task2_local"
    task3_reference_dir: Path | None = None
    task3_output_limit: int = 0
    task3_modes: tuple[str, ...] = ("standard",)
    task3_models: tuple[str, ...] = ("nb2",)
    task3_targets: tuple[str, ...] = ("1K",)
    task4_mockup_engine: str = "direct_ai"
    task4_background: str = ""
    task4_ai_limit: int = 5
    task4_variants_per_product: int = 5
    task4_quality_attempts: int = 3
    template_mockup_model: str = "gemini-3-pro-image"
    task4_modes: tuple[str, ...] = ("flex",)
    task4_models: tuple[str, ...] = ("pro",)
    task4_final_integration: str = "on"
    crop_mode: str = "cover"
    dedupe_threshold: int = 6
    remove_white_background: bool = False
    export_cmyk: bool = True
    mockup_count: int = 5


def product_preset(name: str) -> ProductTarget:
    normalized = name.lower().strip()
    if normalized == "blanket":
        return ProductTarget(name="blanket", width_px=10000, height_px=11000)
    if normalized == "custom":
        return ProductTarget(name="custom", width_px=4000, height_px=6400, allow_custom_shape=True)
    return ProductTarget(name="rug", width_px=4000, height_px=6400)


def restore_pipeline_config(raw_cfg: dict, fallback_root: Path) -> PipelineConfig:
    import dataclasses
    target_raw = raw_cfg.get("target") or {}
    target_obj = ProductTarget(
        name=str(target_raw.get("name", "rug")),
        width_px=int(target_raw.get("width_px", 4000)),
        height_px=int(target_raw.get("height_px", 6400)),
        dpi=int(target_raw.get("dpi", 300)),
        prefer_cmyk=bool(target_raw.get("prefer_cmyk", True)),
        allow_custom_shape=bool(target_raw.get("allow_custom_shape", False)),
        rug_shape=str(target_raw.get("rug_shape", "rectangle")),
    )
    valid_fields = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    kwargs: dict[str, object] = {}
    for key, val in raw_cfg.items():
        if key == "target" or key not in valid_fields:
            continue
        field_type = str(valid_fields[key].type)
        if "Path" in field_type:
            kwargs[key] = Path(val) if val else None
        elif "tuple" in field_type and isinstance(val, list):
            kwargs[key] = tuple(val)
        else:
            kwargs[key] = val
    kwargs["target"] = target_obj
    if "output_root" not in kwargs or not kwargs["output_root"]:
        kwargs["output_root"] = fallback_root
    return PipelineConfig(**kwargs)
