from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComparisonRow:
    index: int
    source_path: Path | None
    cutout_path: Path | None
    cutout_white_path: Path | None
    final_print_path: Path | None
    ai_background_path: Path | None
    # Keep report consumers written before the multi-view gallery compatible.
    ai_background_paths: tuple[Path, ...] = field(default_factory=tuple, kw_only=True)
    status: str
    reason: str
    product_label: str


def build_comparison_rows(run_dir: Path, stage_manifest: dict[str, Any] | None) -> list[ComparisonRow]:
    if not isinstance(stage_manifest, dict):
        return []

    asset_records = [item for item in stage_manifest.get("product_asset_records", []) if isinstance(item, dict)]
    design_records = [item for item in stage_manifest.get("design_records", []) if isinstance(item, dict)]
    cutout_records = [item for item in stage_manifest.get("product_cutout_records", []) if isinstance(item, dict)]
    product_render_records = [item for item in stage_manifest.get("product_render_records", []) if isinstance(item, dict)]
    task4_results = [item for item in stage_manifest.get("task4_results", []) if isinstance(item, dict)]
    ai_final_records = [item for item in stage_manifest.get("ai_background_final_records", []) if isinstance(item, dict)]

    asset_to_cutout: dict[str, tuple[Path | None, Path | None]] = {}
    for record in cutout_records:
        source = resolve_run_path(record.get("source_path"), run_dir)
        if source is None:
            continue
        asset_to_cutout[path_key(source)] = (
            resolve_run_path(record.get("output_path"), run_dir),
            resolve_run_path(record.get("white_output_path"), run_dir),
        )

    asset_to_final = final_print_by_design_source(run_dir, design_records)
    asset_to_ai = ai_background_by_product(run_dir, task4_results, ai_final_records, product_render_records)

    rows: list[ComparisonRow] = []
    for index, record in enumerate(asset_records, start=1):
        source = resolve_run_path(record.get("source_path"), run_dir)
        asset = resolve_run_path(record.get("asset_path"), run_dir)
        cutout, cutout_white = asset_to_cutout.get(path_key(asset), (asset, None)) if asset else (None, None)
        if cutout_white is None and asset is not None:
            white_candidate = run_dir / "product_cutouts_white" / asset.name
            cutout_white = white_candidate if white_candidate.exists() else None
        profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
        profile_final = resolve_run_path(profile.get("final_print_path"), run_dir)
        background_paths = tuple(asset_to_ai.get(path_key(asset), [])) if asset else ()
        rows.append(
            ComparisonRow(
                index=index,
                source_path=source,
                cutout_path=cutout,
                cutout_white_path=cutout_white,
                final_print_path=(asset_to_final.get(path_key(asset)) if asset else None) or profile_final,
                ai_background_path=background_paths[0] if background_paths else None,
                ai_background_paths=background_paths,
                status=str(record.get("status") or ""),
                reason=str(record.get("reason") or ""),
                product_label=str(profile.get("product_label") or ""),
            )
        )
    return rows


def final_print_by_design_source(run_dir: Path, design_records: list[dict[str, Any]]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for record in design_records:
        source = resolve_run_path(record.get("source_path"), run_dir)
        design_path = resolve_run_path(record.get("output_path"), run_dir)
        if source is None or design_path is None:
            continue
        final_path = final_print_path_for_design(run_dir, design_path)
        if final_path is not None:
            output[path_key(source)] = final_path
    return output


def final_print_path_for_design(run_dir: Path, design_path: Path) -> Path | None:
    name = design_path.name
    if not name.endswith("_design.png"):
        return None
    base = name[: -len("_design.png")]
    matches = sorted((run_dir / "final_print").glob(f"{base}_*_rgb.png"))
    return matches[0] if matches else None


def ai_background_by_product(
    run_dir: Path,
    task4_results: list[dict[str, Any]],
    ai_final_records: list[dict[str, Any]],
    product_render_records: list[dict[str, Any]],
) -> dict[str, list[Path]]:
    print_to_product: dict[str, Path] = {}
    for record in product_render_records:
        print_path = resolve_run_path(record.get("print_path"), run_dir)
        product_path = resolve_run_path(record.get("product_path"), run_dir)
        if print_path is not None and product_path is not None:
            print_to_product[path_key(print_path)] = product_path

    run_dir_to_product: dict[str, Path] = {}
    for record in task4_results:
        product = resolve_run_path(record.get("product_path"), run_dir)
        task_run = resolve_run_path(record.get("run_dir"), run_dir)
        if product is not None and task_run is not None:
            run_dir_to_product[path_key(task_run)] = product

    output: dict[str, list[Path]] = {}
    for record in ai_final_records:
        # template_ai records preserve the print-to-product contract directly.
        template_print = resolve_run_path(record.get("print_path"), run_dir)
        template_mockup = resolve_run_path(record.get("mockup_path"), run_dir)
        if template_print is not None and template_mockup is not None:
            matched_product = print_to_product.get(path_key(template_print))
            if matched_product is not None:
                output.setdefault(path_key(matched_product), []).append(template_mockup)
            continue

        source = resolve_run_path(record.get("source_path"), run_dir)
        final_path = resolve_run_path(record.get("final_rgb_path"), run_dir)
        if source is None or final_path is None:
            continue
        matched_product = None
        for task_run_key, product in run_dir_to_product.items():
            if path_key(source).startswith(task_run_key):
                matched_product = product
                break
        if matched_product is not None:
            output.setdefault(path_key(matched_product), []).append(final_path)
    return output


def resolve_run_path(value: object, run_dir: Path) -> Path | None:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            resolved = resolve_run_path(item, run_dir)
            if resolved is not None:
                return resolved
        return None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [run_dir / path, Path.cwd() / path]
    parts = path.parts
    if run_dir.name in parts:
        index = parts.index(run_dir.name)
        candidates.append(run_dir.joinpath(*parts[index + 1 :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def path_key(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()
