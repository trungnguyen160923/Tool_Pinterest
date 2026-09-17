from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable

from PIL import Image, ImageDraw

from .config import PipelineConfig
from .crawler import CandidateImage
from .dedupe import DedupeDecision, dedupe_candidates, is_low_information
from .design import make_print_design
from .artwork_generation import generate_flat_artwork
from .blender_renderer import build_blender_mockup
from .enhancement import enhance_for_print
from .image_ops import export_cmyk_jpg, fit_to_target, remove_near_white_background
from .mockup_profile import mockup_profile_for_target
from .network import blocked_endpoint_summary, check_https_endpoints
from .product import make_product_mockups
from .product_asset import ProductAssetConfig, extract_product_assets
from .product_render import ProductRenderRecord, render_product_from_print
from .printability import assess_candidate, assess_final_artwork, assess_generated_artwork, assess_product_mockup, reference_design_brief
from .report import write_json, write_report
from .rug_shape import recommend_rug_shape
from .task3_adapter import Task3ReplacementConfig, run_task3_replacements
from .task4_adapter import Task4MockupConfig, Task4MockupResult, run_one_task4_mockup, run_task4_mockups
from .template_mockup import build_direct_ai_mockup, build_template_mockup, template_pose_for_index
from .task5_adapter import (
    Task5CrawlerConfig,
    Task5TrendConfig,
    candidates_from_hot_product_images,
    is_task5_auth_error,
    queries_from_trend_package,
    run_task5_image_crawler,
    run_task5_trend_finder,
)
from .task6_adapter import Task6SimilarityConfig, reject_matches_from_task6


ProgressLogger = Callable[[str], None]


class PipelineCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    report_path: Path
    kept_images: list[Path]
    rejected_images: list[Path]
    final_images: list[Path]
    mockups: list[Path]


@dataclass(frozen=True)
class CandidateReviewItem:
    image_id: str
    local_path: str
    image_url: str
    pin_url: str
    pin_id: str
    trend: str
    query: str
    image_score: float
    flat_artwork_score: float
    printability_score: float
    classification: str
    is_direct_printable: bool
    recommended: bool
    width: int | None
    height: int | None
    reason: str
    motifs: list[str]
    source_role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateReviewPackage:
    run_dir: Path
    candidates: list[CandidateReviewItem]
    trend_package_path: Path
    crawl_dir: Path
    config: PipelineConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "trend_package_path": str(self.trend_package_path),
            "crawl_dir": str(self.crawl_dir),
            "candidate_count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def write_resilient_report(
    path: Path,
    config: PipelineConfig,
    decisions: list[DedupeDecision],
    final_images: list[Path],
    mockups: list[Path],
    stage_manifest: dict[str, object],
) -> Path:
    """A presentation failure must never invalidate completed production assets."""
    try:
        return write_report(path, config, decisions, final_images, mockups, stage_manifest)
    except Exception as exc:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "<!doctype html><title>Report unavailable</title>"
            "<h1>Production assets completed</h1>"
            f"<p>Report rendering failed: {str(exc)!r}</p>"
            "<p>Use stage_manifest.json and the output folders to access the generated assets.</p>",
            encoding="utf-8",
        )
        return path


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressLogger | None = None,
    cancel_event: Event | None = None,
) -> PipelineResult:
    base_progress = progress

    def guarded_progress(message: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Find and build was stopped by the user.")
        if base_progress:
            base_progress(message)

    progress = guarded_progress
    log(progress, "Starting product workflow.")

    mode = normalize_key(config.workflow_mode)
    if mode in {"trend_to_product", "auto", "default", ""}:
        package = run_crawl_and_review_stage(config, progress=progress, cancel_event=cancel_event)
        selected = [c for c in package.candidates if c.recommended]
        if not selected:
            selected = package.candidates[: max(1, config.desired_output_count)]
        if not selected:
            raise RuntimeError("No candidate images passed crawl & review.")
        return run_production_from_candidates(
            selected_items=selected,
            config=config,
            run_dir=package.run_dir,
            progress=progress,
            cancel_event=cancel_event,
        )

    run_dir = make_run_dir(config.output_root)
    log(progress, f"Created run folder: {run_dir}")
    kept_dir = run_dir / "dedupe" / "kept"
    rejected_dir = run_dir / "dedupe" / "rejected"
    design_dir = run_dir / "designs"
    enhanced_dir = run_dir / "enhanced"
    cropped_dir = run_dir / "cropped"
    final_dir = run_dir / "final_print"
    product_cutout_dir = run_dir / "product_cutouts"
    mockup_dir = run_dir / "mockups"
    final_dir.mkdir(parents=True, exist_ok=True)

    config, task5_candidates = prepare_discovery(config, run_dir, progress, cancel_event=cancel_event)
    log(progress, "Writing config.json.")
    write_json(run_dir / "config.json", asdict(config))

    log(progress, "Collecting candidate images from Pinterest Browser crawl.")
    candidates = collect_candidates(task5_candidates)
    log(progress, f"Collected {len(candidates)} candidate image(s).")
    if not candidates:
        stage_manifest = {
            "status": "failed",
            "reason": "no_candidate_images",
            "message": (
                "No candidate images were collected. Check Pinterest crawler logs and network/firewall access."
            ),
            "design_records": [],
            "enhancement_records": [],
            "task3_results": [],
            "task4_results": [],
        }
        write_json(run_dir / "stage_manifest.json", stage_manifest)
        write_resilient_report(run_dir / "report.html", config, [], [], [], stage_manifest)
        raise RuntimeError(stage_manifest["message"])
    if config.task6_gallery:
        log(progress, f"Running similarity check in {config.task6_mode} mode.")
    task6_filtered, task6_decisions = reject_matches_from_task6(candidates, task6_config(config))
    log(progress, f"Similarity check rejected {len(task6_decisions)} image(s).")
    log(progress, "Filtering unreadable or low-information images.")
    filtered, filter_decisions = filter_candidates(task6_filtered)
    log(progress, f"Quality filter rejected {len(filter_decisions)} image(s).")
    log(progress, f"Running perceptual dedupe with threshold {config.dedupe_threshold}.")
    kept, dedupe_decisions = dedupe_candidates(filtered, config.dedupe_threshold)
    decisions = task6_decisions + filter_decisions + dedupe_decisions
    log(progress, f"Dedupe kept {len(kept)} image(s), rejected {len([item for item in dedupe_decisions if not item.kept])}.")
    if not kept:
        stage_manifest = {
            "status": "failed",
            "reason": "no_images_after_filtering",
            "message": "All candidate images were rejected by similarity check, quality filter, or dedupe.",
            "design_records": [],
            "enhancement_records": [],
            "task3_results": [],
            "task4_results": [],
        }
        write_json(run_dir / "stage_manifest.json", stage_manifest)
        write_resilient_report(run_dir / "report.html", config, decisions, [], [], stage_manifest)
        raise RuntimeError(stage_manifest["message"])

    log(progress, "Copying kept/rejected decision files.")
    copy_decision_files(decisions, kept_dir, rejected_dir)

    if normalize_key(config.workflow_mode) == "trend_to_product":
        return run_trend_to_product_pipeline(config, run_dir, kept, decisions, progress)

    product_asset_records = []
    product_asset_records_raw = []
    product_asset_record_by_source = {}
    extract_assets_before_print = should_extract_product_assets_before_print(config)
    if extract_assets_before_print:
        log(
            progress,
            f"Extracting clean product assets from {len(kept)} candidate image(s) before print export.",
        )
        _, product_asset_records_raw = extract_product_assets(
            kept,
            product_asset_config(config, run_dir / "product_assets"),
            progress=progress,
        )
        product_asset_records = [record.to_dict() for record in product_asset_records_raw]
        product_asset_record_by_source = {
            path_key(record.source_path): record
            for record in product_asset_records_raw
        }
        if config.product_asset_mode == "required" and not any(
            record.status == "accepted" and record.asset_path is not None
            for record in product_asset_records_raw
        ):
            raise RuntimeError("No clean product assets passed quality gates.")

    final_images: list[Path] = []
    final_pngs: list[Path] = []
    final_png_by_source: dict[str, Path] = {}
    design_records = []
    enhancement_records = []
    product_cutout_records = []
    ai_background_final_records = []
    for index, candidate in enumerate(kept, start=1):
        if len(final_pngs) >= max(1, config.desired_output_count):
            break
        design_source = candidate.path
        asset_record = product_asset_record_by_source.get(path_key(candidate.path))
        if extract_assets_before_print:
            if asset_record is None or asset_record.status != "accepted" or asset_record.asset_path is None:
                log(
                    progress,
                    f"[{index}/{len(kept)}] Skipping print export for {candidate.path.name}: no clean product asset.",
                )
                continue
            design_source = asset_record.asset_path

        output_index = len(final_pngs) + 1
        base = f"{config.target.name}_{output_index:03d}"
        log(progress, f"[{index}/{len(kept)}] Creating print design from {design_source.name}.")
        design_record = make_print_design(design_source, design_dir / f"{base}_design.png", config.target, config.design_mode)
        design_records.append(design_record.to_dict())
        log(progress, f"[{index}/{len(kept)}] Enhancing/upscaling {design_record.output_path.name}.")
        enhancement_record = enhance_for_print(
            design_record.output_path,
            enhanced_dir / f"{base}_enhanced.png",
            min_long_edge=max(config.target.width_px, config.target.height_px),
            mode=config.enhancement_mode,
        )
        enhancement_records.append(enhancement_record.to_dict())
        enhanced = enhancement_record.output_path
        source_for_crop = enhanced
        if config.remove_white_background:
            log(progress, f"[{index}/{len(kept)}] Removing near-white background.")
            source_for_crop = remove_near_white_background(enhanced, enhanced_dir / f"{base}_transparent.png")
        log(progress, f"[{index}/{len(kept)}] Fitting to {config.target.width_px}x{config.target.height_px}.")
        cropped = fit_to_target(source_for_crop, cropped_dir / f"{base}_{config.target.width_px}x{config.target.height_px}.png", config.target, config.crop_mode)
        final_png = final_dir / f"{base}_{config.target.width_px}x{config.target.height_px}_{config.target.dpi}dpi_rgb.png"
        final_png.write_bytes(cropped.read_bytes())
        final_images.append(final_png)
        final_pngs.append(final_png)
        final_png_by_source[path_key(candidate.path)] = final_png
        if config.export_cmyk:
            log(progress, f"[{index}/{len(kept)}] Exporting CMYK JPG.")
            final_images.append(export_cmyk_jpg(final_png, final_dir / f"{base}_{config.target.width_px}x{config.target.height_px}_{config.target.dpi}dpi_cmyk.jpg", config.target.dpi))

    mockups: list[Path] = []
    task4_results = []
    task4_skipped_records = []
    if final_pngs:
        first_png = final_pngs[0]
        log(progress, "Rendering local semantic mockups.")
        mockups = make_product_mockups(first_png, mockup_dir, config.target, config.mockup_count)
    if config.task4_mockup_engine == "task4_ai" and config.task4_ai_limit > 0:
        source_candidates = kept if config.product_asset_mode != "off" else kept[: config.task4_ai_limit]
        task4_input_label = "clean product asset"
        task4_uses_print_ready_fallback = False
        if config.product_asset_mode == "off":
            log(progress, "Product asset extraction is off; Task4 will use original crawled images.")
            source_products = [candidate.path for candidate in source_candidates]
            mask_files = {}
            task4_input_label = "original crawled image"
        else:
            if not product_asset_records_raw:
                log(
                    progress,
                    f"Extracting clean product assets from {len(source_candidates)} candidate image(s) "
                    f"to find up to {config.task4_ai_limit} AI background input(s).",
                )
                _, product_asset_records_raw = extract_product_assets(
                    source_candidates,
                    product_asset_config(config, run_dir / "product_assets"),
                    progress=progress,
                )
                product_asset_records = [record.to_dict() for record in product_asset_records_raw]
            source_products = []
            mask_files = {}
            selected_cutout_sources = []
            count = min(len(source_candidates), len(product_asset_records_raw))
            for item_index in range(count):
                if len(source_products) >= config.task4_ai_limit:
                    break
                record = product_asset_records_raw[item_index]
                if record.status == "accepted" and record.asset_path is not None and record.mask_path is not None:
                    source_products.append(record.asset_path)
                    selected_cutout_sources.append(record.asset_path)
                    mask_files[record.asset_path] = record.mask_path
                else:
                    task4_skipped_records.append(
                        {
                            "source_path": source_candidates[item_index].path,
                            "final_print_path": final_png_by_source.get(path_key(source_candidates[item_index].path)),
                            "stage": "task4_ai_background",
                            "reason": record.reason,
                            "status": "skipped_missing_product_cutout",
                        }
                    )
            product_cutout_records.extend(
                copy_product_cutouts(
                    selected_cutout_sources,
                    product_cutout_dir,
                    source_label="product_asset",
                )
            )
            if not source_products and config.product_asset_mode == "required":
                raise RuntimeError("No clean product assets passed quality gates.")
            if task4_skipped_records:
                log(progress, f"Skipped {len(task4_skipped_records)} image(s) because no product cutout was available.")
        log(progress, f"Running AI background replacement for {len(source_products)} {task4_input_label}(s).")
        if not source_products:
            log(progress, "No Task4 input images are available; skipping AI background replacement.")
            task4_results_raw = []
        else:
            task4_results_raw = run_task4_mockups(
                source_products,
                task4_config(config, run_dir / "task4_mockups"),
                progress=progress,
                mask_files=mask_files,
            )
        task4_results = [result.to_dict() for result in task4_results_raw]
        task4_failed = [result for result in task4_results_raw if result.status != "ok"]
        if task4_failed:
            log(progress, f"AI background replacement failed for {len(task4_failed)} / {len(task4_results_raw)} image(s).")
        for result in task4_results_raw:
            mockups.extend(result.outputs)
        if not product_cutout_records and not task4_uses_print_ready_fallback:
            product_cutout_records.extend(collect_task4_product_cutouts(task4_results_raw, product_cutout_dir))
        elif task4_uses_print_ready_fallback:
            log(progress, "Skipping Task4 product_cutout export because Task4 used print-ready fallback masks.")
        log(progress, "Preparing AI background lifestyle assets.")
        ai_final_paths, ai_background_final_records = prepare_ai_background_assets(
            task4_results_raw,
            config,
            run_dir,
            progress,
        )
        # Lifestyle mockups are previews/listing assets, never print masters.

    task3_results = []
    if final_pngs and config.task3_reference_dir and config.task3_output_limit > 0:
        log(progress, f"Running AI artwork replacement for {config.task3_output_limit} design(s).")
        task3_results_raw = run_task3_replacements(final_pngs, task3_config(config, run_dir / "task3_replacements"))
        task3_results = [result.to_dict() for result in task3_results_raw]

    stage_manifest = {
        "design_records": design_records,
        "enhancement_records": enhancement_records,
        "product_asset_records": product_asset_records,
        "product_cutout_records": product_cutout_records,
        "ai_background_final_records": ai_background_final_records,
        "task4_skipped_records": task4_skipped_records,
        "task3_results": task3_results,
        "task4_results": task4_results,
    }
    log(progress, "Writing stage_manifest.json.")
    write_json(run_dir / "stage_manifest.json", stage_manifest)

    log(progress, "Writing report.html.")
    report_path = write_resilient_report(run_dir / "report.html", config, decisions, final_images, mockups, stage_manifest)
    log(progress, "Product workflow complete.")
    return PipelineResult(
        run_dir=run_dir,
        report_path=report_path,
        kept_images=[candidate.path for candidate in kept],
        rejected_images=[decision.candidate.path for decision in decisions if not decision.kept],
        final_images=final_images,
        mockups=mockups,
    )


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_crawl_and_review_stage(
    config: PipelineConfig,
    run_dir: Path | None = None,
    progress: ProgressLogger | None = None,
    cancel_event: Event | None = None,
) -> CandidateReviewPackage:
    base_progress = progress

    def guarded_progress(message: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Crawl and review stage was stopped by the user.")
        if base_progress:
            base_progress(message)

    progress = guarded_progress
    log(progress, "Starting Step 1: Crawl & AI Vision Pre-screening.")
    if run_dir is None:
        run_dir = make_run_dir(config.output_root)
    log(progress, f"Run folder: {run_dir}")

    config, task5_candidates = prepare_discovery(config, run_dir, progress, cancel_event=cancel_event)
    write_json(run_dir / "config.json", asdict(config))

    log(progress, f"Collected {len(task5_candidates)} crawled image(s).")
    if not task5_candidates:
        review_manifest = {
            "status": "failed",
            "reason": "no_candidate_images",
            "message": "No candidate images were collected. Check Pinterest crawler logs and network/login access.",
            "candidates": [],
        }
        write_json(run_dir / "candidate_review.json", review_manifest)
        raise RuntimeError(review_manifest["message"])

    filtered, filter_decisions = filter_candidates(task5_candidates)
    kept, dedupe_decisions = dedupe_candidates(filtered, config.dedupe_threshold)
    decisions = filter_decisions + dedupe_decisions

    kept_dir = run_dir / "dedupe" / "kept"
    rejected_dir = run_dir / "dedupe" / "rejected"
    copy_decision_files(decisions, kept_dir, rejected_dir)

    review_candidates: list[CandidateReviewItem] = []
    for candidate in kept:
        meta = candidate.metadata or {}
        image_id = str(meta.get("image_id") or candidate.path.stem)
        flat_artwork_score = float(meta.get("flat_artwork_score") or 0.0)
        printability_score = float(meta.get("printability_score") or 0.0)
        is_direct = bool(meta.get("is_direct_printable", False))
        if not is_direct and flat_artwork_score >= 0.70 and printability_score >= 0.65:
            is_direct = True
        classification = str(
            meta.get("classification")
            or ("Flat Pattern" if flat_artwork_score >= 0.75 else "Printable Artwork")
        )
        score = float(meta.get("image_score") or 0.0)
        review_candidates.append(
            CandidateReviewItem(
                image_id=image_id,
                local_path=str(candidate.path.resolve()),
                image_url=str(meta.get("image_url") or ""),
                pin_url=str(meta.get("pin_url") or ""),
                pin_id=str(meta.get("pin_id") or ""),
                trend=str(meta.get("trend") or candidate.keyword),
                query=str(meta.get("query") or candidate.keyword),
                image_score=score,
                flat_artwork_score=flat_artwork_score,
                printability_score=printability_score,
                classification=classification,
                is_direct_printable=is_direct,
                recommended=False,
                width=int(meta.get("width") or 0) or None,
                height=int(meta.get("height") or 0) or None,
                reason=str(meta.get("reason") or ""),
                motifs=list(meta.get("motifs") or []),
                source_role=str(candidate.source_role or meta.get("source_role") or "unknown"),
            )
        )

    # Sort review candidates by direct printability first, then image score
    review_candidates.sort(key=lambda c: (1 if c.is_direct_printable else 0, c.image_score), reverse=True)

    # Pre-select top candidates up to desired_output_count
    recommended_count = min(len(review_candidates), max(1, config.desired_output_count))
    final_review_candidates: list[CandidateReviewItem] = []
    for idx, c in enumerate(review_candidates):
        is_rec = idx < recommended_count
        final_review_candidates.append(replace(c, recommended=is_rec))

    review_manifest = {
        "status": "ready_for_review",
        "run_dir": str(run_dir),
        "target_product": config.target.name,
        "target_size": f"{config.target.width_px}x{config.target.height_px}",
        "niche": config.trend_niche,
        "total_candidates": len(final_review_candidates),
        "direct_printable_count": sum(1 for c in final_review_candidates if c.is_direct_printable),
        "candidates": [c.to_dict() for c in final_review_candidates],
    }
    write_json(run_dir / "candidate_review.json", review_manifest)
    log(progress, f"Step 1 Complete: {len(final_review_candidates)} candidate(s) ready for review ({review_manifest['direct_printable_count']} direct printable).")

    return CandidateReviewPackage(
        run_dir=run_dir,
        candidates=final_review_candidates,
        trend_package_path=run_dir / "task5_trends" / "trend_package.json",
        crawl_dir=run_dir / "task5_crawl",
        config=config,
    )


def run_production_from_candidates(
    selected_items: list[Path | str | dict | CandidateReviewItem | CandidateImage],
    config: PipelineConfig,
    run_dir: Path | None = None,
    progress: ProgressLogger | None = None,
    cancel_event: Event | None = None,
) -> PipelineResult:
    base_progress = progress

    def guarded_progress(message: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Production was stopped by the user.")
        if base_progress:
            base_progress(message)

    progress = guarded_progress
    log(progress, f"Starting Step 2: Production for {len(selected_items)} selected image(s).")
    if run_dir is None:
        run_dir = make_run_dir(config.output_root)
    log(progress, f"Production folder: {run_dir}")

    # Resolve candidate image paths and metadata
    resolved_sources: list[tuple[Path, str, dict[str, object]]] = []
    for item in selected_items:
        if isinstance(item, CandidateReviewItem):
            path = Path(item.local_path)
            keyword = item.query or item.trend
            meta = item.to_dict()
        elif isinstance(item, CandidateImage):
            path = item.path
            keyword = item.keyword
            meta = item.metadata or {}
        elif isinstance(item, dict):
            raw_path = str(item.get("local_path") or item.get("path") or "")
            path = Path(raw_path)
            keyword = str(item.get("query") or item.get("trend") or item.get("keyword") or path.stem)
            meta = dict(item)
        elif isinstance(item, (str, Path)):
            path = Path(item)
            keyword = path.stem
            meta = {"local_path": str(path)}
        else:
            continue
        if path.exists() and path.is_file():
            resolved_sources.append((path, keyword, meta))

    if not resolved_sources:
        raise RuntimeError("No valid candidate images provided for production.")

    design_dir = run_dir / "artwork_designs"
    enhanced_dir = run_dir / "enhanced"
    cropped_dir = run_dir / "cropped"
    final_dir = run_dir / "final_print"
    rendered_product_dir = run_dir / "rendered_products"
    rendered_mask_dir = run_dir / "rendered_product_masks"
    product_cutout_dir = run_dir / "product_cutouts"
    mockup_dir = run_dir / "mockups"
    lifestyle_dir = run_dir / "lifestyle_mockups"
    for d in (design_dir, enhanced_dir, cropped_dir, final_dir, rendered_product_dir, rendered_mask_dir, product_cutout_dir, mockup_dir, lifestyle_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "production_config.json", asdict(config))

    final_images: list[Path] = []
    final_pngs: list[Path] = []
    rendered_products: list[Path] = []
    rendered_masks: dict[Path, Path] = {}
    design_records = []
    enhancement_records = []
    artwork_generation_records = []
    product_render_records = []
    product_asset_records = []
    rug_shape_records = []
    template_mockup_records = []
    mockup_quality_records = []
    render_target_by_print: dict[str, object] = {}

    is_direct_mode = normalize_key(config.design_mode) in {"direct", "direct_print", "product_design"}

    for index, (source_path, keyword, meta) in enumerate(resolved_sources, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Production was stopped by the user.")

        base = f"{config.target.name}_{index:03d}"
        log(progress, f"[{index}/{len(resolved_sources)}] Processing design from {source_path.name}.")

        design_source = source_path
        applied_design_mode = "direct"

        if not is_direct_mode:
            generated_path = run_dir / "generated_artwork" / f"{base}_gemini.png"
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            log(progress, f"[{index}/{len(resolved_sources)}] Generating flat artwork with Gemini from reference.")
            generation = generate_flat_artwork(
                source_path,
                generated_path,
                config.target,
                backend=config.gemini_backend,
                image_size=config.artwork_image_size,
            )
            artwork_generation_records.append(generation.to_dict())
            if generation.status == "ok" and generation.output_path and generation.output_path.exists():
                design_source = generation.output_path
            else:
                log(progress, f"[{index}/{len(resolved_sources)}] Gemini redraw failed; using original image directly.")
                design_source = source_path

        design_record = make_print_design(
            design_source,
            design_dir / f"{base}_artwork.png",
            config.target,
            applied_design_mode,
        )
        design_records.append(design_record.to_dict())

        log(progress, f"[{index}/{len(resolved_sources)}] Enhancing/upscaling for print.")
        enhancement_record = enhance_for_print(
            design_record.output_path,
            enhanced_dir / f"{base}_enhanced.png",
            min_long_edge=max(config.target.width_px, config.target.height_px),
            mode=config.enhancement_mode,
        )
        enhancement_records.append(enhancement_record.to_dict())

        source_for_crop = enhancement_record.output_path
        if config.remove_white_background:
            log(progress, f"[{index}/{len(resolved_sources)}] Removing near-white background.")
            source_for_crop = remove_near_white_background(
                enhancement_record.output_path,
                enhanced_dir / f"{base}_transparent.png",
            )

        log(progress, f"[{index}/{len(resolved_sources)}] Fitting canvas to {config.target.width_px}x{config.target.height_px} ({config.crop_mode}).")
        cropped = fit_to_target(
            source_for_crop,
            cropped_dir / f"{base}_{config.target.width_px}x{config.target.height_px}.png",
            config.target,
            config.crop_mode,
        )

        final_png = final_dir / f"{base}_{config.target.width_px}x{config.target.height_px}_{config.target.dpi}dpi_rgb.png"
        final_png.write_bytes(cropped.read_bytes())
        final_images.append(final_png)
        final_pngs.append(final_png)

        if config.export_cmyk:
            log(progress, f"[{index}/{len(resolved_sources)}] Exporting CMYK print JPG.")
            cmyk_path = final_dir / f"{base}_{config.target.width_px}x{config.target.height_px}_{config.target.dpi}dpi_cmyk.jpg"
            final_images.append(export_cmyk_jpg(final_png, cmyk_path, config.target.dpi))

        render_target = config.target
        if config.target.name.strip().lower() == "rug":
            try:
                shape_decision = recommend_rug_shape(
                    final_png,
                    config.target,
                    backend=config.gemini_backend,
                    model=config.gemini_model,
                )
                render_target = replace(config.target, rug_shape=shape_decision.shape)
                rug_shape_records.append({"print_path": final_png, **shape_decision.to_dict()})
                log(progress, f"[{index}/{len(resolved_sources)}] Rug shape: {shape_decision.shape}.")
            except Exception as shape_exc:
                log(progress, f"Rug shape recommendation skipped: {shape_exc}")

        render_target_by_print[path_key(final_png)] = render_target
        try:
            render_record = render_product_from_print(
                source_path=source_path,
                print_path=final_png,
                product_path=rendered_product_dir / f"{base}_product.png",
                mask_path=rendered_mask_dir / f"{base}_mask.png",
                target=render_target,
            )
            product_render_records.append(render_record)
            rendered_products.append(render_record.product_path)
            rendered_masks[render_record.product_path] = render_record.mask_path
            product_asset_records.append(
                {
                    "source_path": source_path,
                    "asset_path": render_record.product_path,
                    "mask_path": render_record.mask_path,
                    "status": "accepted",
                    "reason": "rendered_from_trend_artwork",
                }
            )
        except Exception as render_exc:
            log(progress, f"Product render skipped: {render_exc}")

    product_cutout_records = []
    if rendered_products:
        product_cutout_records = copy_product_cutouts(rendered_products, product_cutout_dir, source_label="rendered_product")

    # Mockups rendering
    mockups: list[Path] = []
    ai_background_final_records: list[dict[str, object]] = []
    if final_pngs:
        log(progress, "Rendering local product mockups.")
        mockups.extend(make_product_mockups(final_pngs[0], mockup_dir, config.target, count=config.mockup_count))

        if config.task4_mockup_engine in {"blender_3d", "direct_ai", "template_ai"} and config.task4_ai_limit > 0:
            source_prints = final_pngs[: config.task4_ai_limit]
            variants_per_product = max(1, config.task4_variants_per_product)
            blender_render = config.task4_mockup_engine == "blender_3d"
            direct_render = config.task4_mockup_engine == "direct_ai"
            for p_idx, print_file in enumerate(source_prints, start=1):
                cur_target = render_target_by_print.get(path_key(print_file), config.target)
                for var_idx in range(1, variants_per_product + 1):
                    pose = template_pose_for_index(cur_target, var_idx)
                    try:
                        if blender_render:
                            rec = build_blender_mockup(print_file, run_dir, cur_target, pose=pose, variant=var_idx, progress=progress)
                        elif direct_render:
                            rec = build_direct_ai_mockup(
                                print_file,
                                run_dir,
                                cur_target,
                                pose=pose,
                                variant=var_idx,
                                backend=config.gemini_backend,
                                model=config.template_mockup_model,
                                quality_model=config.gemini_model,
                                attempts=max(1, config.task4_quality_attempts),
                                progress=progress,
                            )
                        else:
                            rec = build_template_mockup(
                                print_file,
                                run_dir,
                                cur_target,
                                backend=config.gemini_backend,
                                model=config.template_mockup_model,
                                quality_model=config.gemini_model,
                                attempts=max(1, config.task4_quality_attempts),
                                pose=pose,
                                variant=var_idx,
                            )
                        template_mockup_records.append(rec.to_dict())
                        if rec.status == "ok" and rec.mockup_path and rec.mockup_path.exists():
                            mockups.append(rec.mockup_path)
                            lifestyle_copy = lifestyle_dir / f"{cur_target.name}_{p_idx:03d}_lifestyle_{var_idx}.png"
                            lifestyle_copy.write_bytes(rec.mockup_path.read_bytes())
                            ai_background_final_records.append({
                                "source_path": rec.mockup_path,
                                "lifestyle_path": lifestyle_copy,
                                "asset_type": "lifestyle_mockup",
                                "status": "ok",
                            })
                    except Exception as mock_exc:
                        log(progress, f"Mockup view {var_idx} skipped: {mock_exc}")

    stage_manifest = {
        "status": "completed",
        "workflow_mode": "trend_to_product",
        "design_mode": config.design_mode,
        "selected_candidates_count": len(resolved_sources),
        "final_images_count": len(final_images),
        "mockups_count": len(mockups),
        "design_records": design_records,
        "enhancement_records": enhancement_records,
        "product_render_records": [record.to_dict() for record in product_render_records],
        "product_asset_records": product_asset_records,
        "product_cutout_records": product_cutout_records,
        "artwork_generation_records": artwork_generation_records,
        "rug_shape_records": rug_shape_records,
        "ai_background_final_records": ai_background_final_records,
        "template_mockup_records": template_mockup_records,
        "mockup_quality_records": mockup_quality_records,
    }
    write_json(run_dir / "stage_manifest.json", stage_manifest)
    decisions = [
        DedupeDecision(CandidateImage(path=p, source="user_review", keyword=kw), True, "selected_for_production")
        for p, kw, _ in resolved_sources
    ]
    report_path = write_resilient_report(run_dir / "report.html", config, decisions, final_images, mockups, stage_manifest)
    log(progress, f"Production complete. Output: {run_dir}")

    return PipelineResult(
        run_dir=run_dir,
        report_path=report_path,
        kept_images=[p for p, _, _ in resolved_sources],
        rejected_images=[],
        final_images=final_images,
        mockups=mockups,
    )


def run_trend_to_product_pipeline(
    config: PipelineConfig,
    run_dir: Path,
    kept: list[CandidateImage],
    decisions: list[DedupeDecision],
    progress: ProgressLogger | None = None,
    cancel_event: Event | None = None,
) -> PipelineResult:
    return run_production_from_candidates(
        selected_items=kept,
        config=config,
        run_dir=run_dir,
        progress=progress,
        cancel_event=cancel_event,
    )



def prepare_discovery(
    config: PipelineConfig,
    run_dir: Path,
    progress: ProgressLogger | None = None,
    *,
    cancel_event: Event | None = None,
) -> tuple[PipelineConfig, list[CandidateImage]]:
    browser_config = replace(config, task5_provider="pinterest-browser")

    if config.network_preflight:
        log(progress, "Checking outbound HTTPS access for Pinterest.")
        checks = check_https_endpoints(["www.pinterest.com"])
        blocked = blocked_endpoint_summary(checks)
        if blocked:
            log(progress, blocked)
            write_json(run_dir / "config.json", asdict(browser_config))
            stage_manifest = {
                "status": "failed",
                "reason": "network_preflight_failed",
                "message": blocked,
                "endpoint_checks": [check.__dict__ for check in checks],
                "design_records": [],
                "enhancement_records": [],
                "task3_results": [],
                "task4_results": [],
            }
            write_json(run_dir / "stage_manifest.json", stage_manifest)
            write_resilient_report(run_dir / "report.html", browser_config, [], [], [], stage_manifest)
            raise RuntimeError(blocked + " Allow outbound HTTPS for python.exe/Chromium and retry.")

    config = browser_config
    niche = config.trend_niche.strip()
    if not niche:
        raise RuntimeError("A Pinterest Trends niche is required.")
    log(progress, f"Finding Pinterest trends for niche '{niche}'.")
    try:
        package_path = run_task5_trend_finder(
            Task5TrendConfig(
                niche=niche,
                output_dir=run_dir / "task5_trends",
                region=config.trend_region,
                trend_type=config.trend_type,
                interest=config.trend_interest,
                keyword_limit=config.trend_keyword_limit,
                max_trends=config.trend_max_trends,
                min_semantic_fit=config.trend_min_semantic_fit,
                gemini_backend=config.gemini_backend,
                gemini_model=config.gemini_model,
                token_path=config.task5_token_path,
                cancel_event=cancel_event,
            ),
            progress=progress,
        )
    except RuntimeError as exc:
        if not is_task5_auth_error(exc):
            raise
        message = f"Pinterest trend discovery failed. Fix Pinterest Trends API credentials/permissions and retry. Details: {exc}"
        log(progress, message)
        raise RuntimeError(message) from exc

    discovered_queries = queries_from_trend_package(package_path, max_queries_per_trend=config.trend_max_queries_per_trend)
    log(progress, f"Loaded {len(discovered_queries)} querie(s) from trend package.")
    log(progress, f"Running Pinterest image crawler with provider '{config.task5_provider}'.")
    hot_images = run_task5_image_crawler(
        Task5CrawlerConfig(
            package_path=package_path,
            output_dir=run_dir / "task5_crawl",
            provider=config.task5_provider,
            max_images_per_query=config.task5_max_images_per_query,
            max_trends=config.task5_max_crawl_trends,
            max_queries_per_trend=config.trend_max_queries_per_trend,
            max_downloads=config.task5_max_downloads,
            top_images=config.task5_top_images,
            vision_mode=config.task5_vision_mode,
            crawl_purpose=task5_crawl_purpose(config),
            product_focus=task5_product_focus(config),
            gemini_backend=config.gemini_backend,
            vision_model=config.gemini_model,
            refresh_vision_cache=config.task5_refresh_vision_cache,
            cancel_event=cancel_event,
        ),
        progress=progress,
    )
    task5_candidates = candidates_from_hot_product_images(hot_images)
    log(progress, f"Imported {len(task5_candidates)} ranked Pinterest image(s).")
    return config, task5_candidates


def log(progress: ProgressLogger | None, message: str) -> None:
    if progress:
        progress(message)


def collect_candidates(task5_candidates: list[CandidateImage]) -> list[CandidateImage]:
    return list(task5_candidates)


def should_extract_product_assets_before_print(config: PipelineConfig) -> bool:
    if config.product_asset_mode == "off":
        return False
    return config.product_asset_mode == "required" or (
        config.task4_mockup_engine == "task4_ai" and config.task4_ai_limit > 0
    )


def path_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()


def normalize_key(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_")


def create_print_ready_fallback_masks(products: list[Path], output_dir: Path) -> dict[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks: dict[Path, Path] = {}
    for product in products:
        with Image.open(product) as img:
            width, height = img.size
        inset = max(2, round(min(width, height) * 0.02))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle(
            (inset, inset, max(inset, width - inset - 1), max(inset, height - inset - 1)),
            fill=255,
        )
        mask_path = output_dir / f"{product.stem}_mask.png"
        mask.save(mask_path)
        masks[product] = mask_path
    return masks


def print_ready_fallback_products_with_cutouts(
    source_candidates: list[CandidateImage],
    final_pngs: list[Path],
    product_asset_records: list,
    *,
    limit: int,
) -> tuple[list[Path], list[dict[str, object]]]:
    products: list[Path] = []
    skipped: list[dict[str, object]] = []
    count = min(max(0, limit), len(source_candidates), len(final_pngs), len(product_asset_records))
    for index in range(count):
        record = product_asset_records[index]
        if getattr(record, "asset_path", None) is not None:
            products.append(final_pngs[index])
            continue
        skipped.append(
            {
                "source_path": source_candidates[index].path,
                "final_print_path": final_pngs[index],
                "stage": "task4_ai_background",
                "reason": getattr(record, "reason", "missing product cutout"),
                "status": "skipped_missing_product_cutout",
            }
        )
    return products, skipped


def collect_task4_product_cutouts(results: list[Task4MockupResult], output_dir: Path) -> list[dict[str, object]]:
    sources: list[Path] = []
    for result in results:
        if result.run_dir is None:
            continue
        cutout = result.run_dir / "intermediates" / "product_cutout.png"
        if cutout.exists():
            sources.append(cutout)
    return copy_product_cutouts(sources, output_dir, source_label="task4_product_cutout")


def copy_product_cutouts(sources: list[Path], output_dir: Path, *, source_label: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    white_output_dir = output_dir.parent / f"{output_dir.name}_white"
    white_output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for index, source in enumerate(sources, start=1):
        if not source.exists():
            continue
        stem = source.stem
        if stem == "product_cutout":
            parent = source.parent.parent.name if source.parent.parent else f"cutout_{index:03d}"
            stem = f"{parent}_product_cutout"
        name = f"{stem}.png"
        if name in used_names:
            name = f"{stem}_{index:03d}.png"
        used_names.add(name)
        destination = output_dir / name
        destination.write_bytes(source.read_bytes())
        white_destination = white_output_dir / name
        write_white_background_copy(source, white_destination)
        records.append(
            {
                "source_path": source,
                "output_path": destination,
                "white_output_path": white_destination,
                "source_label": source_label,
                "status": "ok",
            }
        )
    return records


def write_white_background_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        white.alpha_composite(image)
        white.convert("RGB").save(destination)


def prepare_ai_background_assets(
    results: list[Task4MockupResult],
    config: PipelineConfig,
    run_dir: Path,
    progress: ProgressLogger | None = None,
) -> tuple[list[Path], list[dict[str, object]]]:
    selected_outputs = task4_background_final_outputs(results)
    final_paths: list[Path] = []
    records: list[dict[str, object]] = []
    if not selected_outputs:
        return final_paths, records

    lifestyle_dir = run_dir / "lifestyle_mockups"
    for index, source in enumerate(selected_outputs, start=1):
        base = f"{config.target.name}_{index:03d}_ai_background"
        log(progress, f"[AI {index}/{len(selected_outputs)}] Saving lifestyle mockup preview.")
        lifestyle_path = lifestyle_dir / f"{base}.png"
        lifestyle_path.parent.mkdir(parents=True, exist_ok=True)
        lifestyle_path.write_bytes(source.read_bytes())
        records.append(
            {
                "source_path": source,
                "lifestyle_path": lifestyle_path,
                "asset_type": "lifestyle_mockup",
                "print_master": False,
                "status": "ok",
            }
        )
    return final_paths, records


def task4_background_final_outputs(results: list[Task4MockupResult]) -> list[Path]:
    outputs: list[Path] = []
    for result in results:
        if result.status != "ok":
            continue
        candidates = [path for path in result.outputs if path.name.endswith("semantic_strict_output.png")]
        if not candidates:
            candidates = [path for path in result.outputs if path.name.endswith("deterministic_composite.png")]
        for path in candidates:
            if path.exists():
                outputs.append(path)
    return outputs


def task6_config(config: PipelineConfig) -> Task6SimilarityConfig | None:
    if not config.task6_gallery:
        return None
    return Task6SimilarityConfig(
        gallery=config.task6_gallery,
        index_path=config.task6_index_path,
        metadata_path=config.task6_metadata_path,
        threshold=config.task6_similarity_threshold,
        rebuild_index=config.task6_rebuild_index,
        mode=config.task6_mode,
        clip_model=config.task6_clip_model,
        device=config.task6_device,
    )


def task3_config(config: PipelineConfig, output_dir: Path) -> Task3ReplacementConfig:
    if config.task3_reference_dir is None:
        raise RuntimeError("Artwork reference dir is required.")
    return Task3ReplacementConfig(
        reference_dir=config.task3_reference_dir,
        output_dir=output_dir,
        modes=config.task3_modes,
        models=config.task3_models,
        targets=config.task3_targets,
        output_limit=config.task3_output_limit,
    )


def task4_config(
    config: PipelineConfig,
    output_dir: Path,
    profile=None,
) -> Task4MockupConfig:
    profile = profile or mockup_profile_for_target(config.target)
    return Task4MockupConfig(
        output_dir=output_dir,
        background=config.task4_background.strip() or automatic_background_brief(config, profile),
        modes=config.task4_modes,
        models=config.task4_models,
        final_integration=config.task4_final_integration,
        limit=config.task4_ai_limit,
    )


def product_asset_config(config: PipelineConfig, output_dir: Path) -> ProductAssetConfig:
    return ProductAssetConfig(
        output_dir=output_dir,
        mode=config.product_asset_mode,
        gemini_backend=config.gemini_backend,
        gemini_model=config.gemini_model,
        target_hint=product_asset_target_hint(config),
        min_visible_percent=config.product_asset_min_visible_percent,
        min_mask_coverage=config.product_asset_min_mask_coverage,
        max_mask_coverage=config.product_asset_max_mask_coverage,
    )


def product_asset_target_hint(config: PipelineConfig) -> str:
    product = config.target.name.strip().lower().replace("_", "-")
    focus = config.task5_product_focus.strip().lower().replace("_", "-")
    if product == "rug" and focus == "area-rug":
        return "area rug"
    if product == "rug":
        return "rug, mat, or floor covering"
    return config.target.name


def task5_product_focus(config: PipelineConfig) -> str:
    focus = config.task5_product_focus.strip().lower().replace("_", "-")
    if focus != "auto":
        return focus
    if task5_crawl_purpose(config) == "inspiration":
        return config.target.name.strip().lower() or "product"
    product = config.target.name.strip().lower().replace("_", "-")
    if product == "blanket":
        return "blanket"
    if product == "rug":
        return "any-floor-covering"
    return "auto"


def task5_crawl_purpose(config: PipelineConfig) -> str:
    purpose = normalize_key(config.task5_crawl_purpose)
    if purpose in {"product", "inspiration"}:
        return purpose
    return "inspiration" if normalize_key(config.workflow_mode) == "trend_to_product" else "product"


def automatic_background_brief(config: PipelineConfig, profile=None) -> str:
    product = config.target.name.strip() or "product"
    context = config.trend_niche.strip() or product
    subject = product if context.lower() == product.lower() else f"{context}-inspired {product}"
    profile = profile or mockup_profile_for_target(config.target)
    return (
        f"Photoreal ecommerce lifestyle background for a {subject}. "
        f"{profile.prompt_contract()} "
        "Keep the original product as the hero subject, preserve its shape and artwork, "
        "use realistic camera perspective, physically correct contact with its support surface, natural contact shadows, "
        "coherent material texture, and clean commercial composition with no text or logos."
    )


def filter_candidates(candidates: list[CandidateImage]) -> tuple[list[CandidateImage], list[DedupeDecision]]:
    accepted: list[CandidateImage] = []
    decisions: list[DedupeDecision] = []
    for candidate in candidates:
        try:
            if is_low_information(candidate.path):
                decisions.append(DedupeDecision(candidate, False, "low_information"))
                continue
        except Exception as exc:
            decisions.append(DedupeDecision(candidate, False, f"unreadable: {exc}"))
            continue
        accepted.append(candidate)
    return accepted, decisions


def copy_decision_files(decisions: list[DedupeDecision], kept_dir: Path, rejected_dir: Path) -> None:
    kept_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for decision in decisions:
        destination = kept_dir if decision.kept else rejected_dir
        target = destination / decision.candidate.path.name
        if not target.exists():
            target.write_bytes(decision.candidate.path.read_bytes())




