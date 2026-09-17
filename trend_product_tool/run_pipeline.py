from __future__ import annotations

import argparse
from pathlib import Path

from trend_tool.config import PipelineConfig, ProductTarget, product_preset
from trend_tool.pipeline import run_pipeline
from trend_tool.settings import env, env_float, env_int, load_tool_env, task5_token_path_from_env


def parse_args() -> argparse.Namespace:
    load_tool_env()
    parser = argparse.ArgumentParser(description="Run the trend product image workflow.")
    parser.add_argument("--output-root", type=Path, default=Path(env("TREND_PRODUCT_OUTPUT", "trend_product_tool/output")))
    parser.add_argument("--workflow-mode", choices=["trend-to-product", "product-crawl"], default="trend-to-product")
    parser.add_argument("--niche", default=env("PINTEREST_NICHE", ""), help="Required Pinterest Trends niche. It is sent to the API unchanged.")
    parser.add_argument("--trend-region", default=env("PINTEREST_REGION", "US"))
    parser.add_argument("--trend-type", choices=["growing", "monthly", "yearly", "seasonal"], default=env("PINTEREST_TREND_TYPE", "growing"))
    parser.add_argument("--trend-interest", default=env("PINTEREST_INTEREST", ""))
    parser.add_argument("--trend-keyword-limit", type=int, default=env_int("TREND_KEYWORD_LIMIT", 50))
    parser.add_argument("--trend-max-trends", type=int, default=env_int("TREND_MAX_TRENDS", 20))
    parser.add_argument("--trend-min-semantic-fit", type=float, default=env_float("TREND_MIN_SEMANTIC_FIT", 35.0))
    parser.add_argument("--trend-max-queries-per-trend", type=int, default=6)
    parser.add_argument("--desired-output-count", type=int, default=5, help="Number of approved artwork outputs to produce.")
    parser.add_argument("--gemini-backend", choices=["auto", "enterprise", "api-key"], default=env("GEMINI_BACKEND", "auto"))
    parser.add_argument("--gemini-model", default=env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--task5-token-path", type=Path, default=task5_token_path_from_env(), help="Optional Pinterest OAuth token JSON path for trend discovery.")
    parser.add_argument("--task5-provider", choices=["pinterest-browser"], default="pinterest-browser")
    parser.add_argument("--task5-max-images-per-query", type=int, default=30)
    parser.add_argument("--task5-max-crawl-trends", type=int, default=5)
    parser.add_argument("--task5-max-downloads", type=int, default=250)
    parser.add_argument("--task5-top-images", type=int, default=100)
    parser.add_argument("--task5-vision-mode", choices=["auto", "required", "off"], default="auto")
    parser.add_argument("--task5-crawl-purpose", choices=["auto", "product", "inspiration"], default="auto")
    parser.add_argument("--task5-product-focus", default="auto", help="Target product focus. Can be auto or any product phrase, e.g. blanket, leather-bag, ceramic-mug.")
    parser.add_argument("--task5-refresh-vision-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--product", choices=["rug", "blanket", "custom"], default="rug")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--crop-mode", choices=["none", "contain", "cover"], default="cover")
    parser.add_argument("--dedupe-threshold", type=int, default=6)
    parser.add_argument("--remove-white-bg", action="store_true")
    parser.add_argument("--export-cmyk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task6-gallery", type=Path, default=None, help="Optional existing gallery to reject visually similar matches.")
    parser.add_argument("--task6-index", type=Path, default=None, help="Optional similarity index path.")
    parser.add_argument("--task6-metadata", type=Path, default=None, help="Optional similarity metadata JSON.")
    parser.add_argument("--task6-threshold", type=float, default=92.0, help="Reject candidates at or above this similarity score.")
    parser.add_argument("--task6-rebuild-index", action="store_true")
    parser.add_argument("--task6-mode", choices=["auto", "clip", "classic"], default="auto")
    parser.add_argument("--task6-clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--task6-device", default="auto")
    parser.add_argument("--product-asset-mode", choices=["auto", "required", "off"], default="auto")
    parser.add_argument("--product-asset-min-visible-percent", type=float, default=80.0)
    parser.add_argument("--product-asset-min-mask-coverage", type=float, default=0.01)
    parser.add_argument("--product-asset-max-mask-coverage", type=float, default=0.70)
    parser.add_argument("--design-mode", choices=["ai-artwork", "direct", "product-design", "pattern-repeat"], default="ai-artwork")
    parser.add_argument("--artwork-image-size", choices=["1K", "2K", "4K"], default=env("GEMINI_ARTWORK_IMAGE_SIZE", "2K"))
    parser.add_argument("--enhancement-mode", choices=["off", "task2-local"], default="task2-local")
    parser.add_argument("--task3-reference-dir", type=Path, default=None, help="Optional product reference folder for AI artwork replacement outputs.")
    parser.add_argument("--task3-output-limit", type=int, default=0, help="Number of final PNG designs to send through AI artwork replacement. 0 disables.")
    parser.add_argument("--task3-modes", nargs="+", choices=["standard", "flex"], default=["standard"])
    parser.add_argument("--task3-models", nargs="+", choices=["lite", "nb2", "pro"], default=["nb2"])
    parser.add_argument("--task3-targets", nargs="+", choices=["1K", "2K", "4K"], default=["1K"])
    parser.add_argument("--task4-mockup-engine", choices=["local-semantic", "task4-ai", "template-ai", "direct-ai", "blender-3d"], default="direct-ai")
    parser.add_argument("--task4-background", default="")
    parser.add_argument("--task4-ai-limit", type=int, default=None, help="Number of product outputs to send through AI background replacement. Defaults to desired output count.")
    parser.add_argument("--task4-variants-per-product", type=int, default=4, help="Number of AI lifestyle views to generate for each product output.")
    parser.add_argument("--task4-modes", nargs="+", choices=["standard", "flex"], default=["flex"])
    parser.add_argument("--task4-models", nargs="+", choices=["nb2", "pro"], default=["pro"])
    parser.add_argument("--task4-final-integration", choices=["on", "off"], default="on")
    parser.add_argument("--template-mockup-model", default="gemini-3-pro-image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = product_preset(args.product)
    if args.width and args.height:
        target = ProductTarget(
            name=args.product,
            width_px=args.width,
            height_px=args.height,
            dpi=args.dpi,
            prefer_cmyk=target.prefer_cmyk,
            allow_custom_shape=target.allow_custom_shape,
        )

    trend_niche = args.niche.strip()
    if not trend_niche:
        raise SystemExit("--niche or PINTEREST_NICHE is required for Pinterest Trends discovery.")
    config = PipelineConfig(
        target=target,
        output_root=args.output_root,
        workflow_mode=args.workflow_mode.replace("-", "_"),
        trend_niche=trend_niche,
        trend_region=args.trend_region,
        trend_type=args.trend_type,
        trend_interest=args.trend_interest,
        trend_keyword_limit=args.trend_keyword_limit,
        trend_max_trends=args.trend_max_trends,
        trend_min_semantic_fit=args.trend_min_semantic_fit,
        trend_max_queries_per_trend=args.trend_max_queries_per_trend,
        desired_output_count=max(1, args.desired_output_count),
        gemini_backend=args.gemini_backend,
        gemini_model=args.gemini_model,
        task5_token_path=args.task5_token_path,
        task5_provider=args.task5_provider,
        task5_max_images_per_query=args.task5_max_images_per_query,
        task5_max_crawl_trends=args.task5_max_crawl_trends,
        task5_max_downloads=args.task5_max_downloads,
        task5_top_images=args.task5_top_images,
        task5_vision_mode=args.task5_vision_mode,
        task5_crawl_purpose=args.task5_crawl_purpose,
        task5_product_focus=args.task5_product_focus,
        task5_refresh_vision_cache=args.task5_refresh_vision_cache,
        crop_mode=args.crop_mode,
        dedupe_threshold=args.dedupe_threshold,
        remove_white_background=args.remove_white_bg,
        export_cmyk=args.export_cmyk,
        task6_gallery=args.task6_gallery,
        task6_index_path=args.task6_index,
        task6_metadata_path=args.task6_metadata,
        task6_similarity_threshold=args.task6_threshold,
        task6_rebuild_index=args.task6_rebuild_index,
        task6_mode=args.task6_mode,
        task6_clip_model=args.task6_clip_model,
        task6_device=args.task6_device,
        product_asset_mode=args.product_asset_mode,
        product_asset_min_visible_percent=args.product_asset_min_visible_percent,
        product_asset_min_mask_coverage=args.product_asset_min_mask_coverage,
        product_asset_max_mask_coverage=args.product_asset_max_mask_coverage,
        design_mode=args.design_mode.replace("-", "_"),
        artwork_image_size=args.artwork_image_size,
        enhancement_mode=args.enhancement_mode.replace("-", "_"),
        task3_reference_dir=args.task3_reference_dir,
        task3_output_limit=args.task3_output_limit,
        task3_modes=tuple(args.task3_modes),
        task3_models=tuple(args.task3_models),
        task3_targets=tuple(args.task3_targets),
        task4_mockup_engine=args.task4_mockup_engine.replace("-", "_"),
        task4_background=args.task4_background,
        task4_ai_limit=max(1, args.task4_ai_limit if args.task4_ai_limit is not None else args.desired_output_count),
        task4_variants_per_product=max(1, min(4, args.task4_variants_per_product)),
        task4_modes=tuple(args.task4_modes),
        task4_models=tuple(args.task4_models),
        task4_final_integration=args.task4_final_integration,
        template_mockup_model=args.template_mockup_model,
    )
    result = run_pipeline(config)
    print(f"Run folder: {result.run_dir}")
    print(f"Report: {result.report_path}")
    print(f"Kept images: {len(result.kept_images)}")
    print(f"Rejected duplicates: {len(result.rejected_images)}")


if __name__ == "__main__":
    main()




