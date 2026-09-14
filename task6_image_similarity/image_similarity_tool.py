#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from image_similarity.embeddings import create_embedder
from image_similarity.gemini_reranker import env as gemini_env
from image_similarity.gemini_reranker import load_env_file, rerank_with_gemini
from image_similarity.indexer import build_index, load_index
from image_similarity.report import export_results
from image_similarity.searcher import query_features, search
from image_similarity.utils import timestamp_slug
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger("task6_image_similarity")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def default_index_path() -> Path:
    return BASE_DIR / "index" / "gallery_index.json"


def default_output_path() -> Path:
    return BASE_DIR / "output" / f"run_{timestamp_slug()}"


def resampling_filter(name: str):
    try:
        return getattr(Image.Resampling, name)
    except AttributeError:
        return getattr(Image, name)


def estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    pixels: list[tuple[int, int, int]] = []

    for x in range(width):
        pixels.append(image.getpixel((x, 0)))
        pixels.append(image.getpixel((x, height - 1)))
    for y in range(height):
        pixels.append(image.getpixel((0, y)))
        pixels.append(image.getpixel((width - 1, y)))

    return (
        sum(pixel[0] for pixel in pixels) // len(pixels),
        sum(pixel[1] for pixel in pixels) // len(pixels),
        sum(pixel[2] for pixel in pixels) // len(pixels),
    )


def estimate_product_bbox(
    image: Image.Image,
    background: tuple[int, int, int],
    threshold: int = 35,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    data = image.load()
    min_x, min_y = width, height
    max_x, max_y = -1, -1

    # Approximate product pixels by comparing each pixel to the border color.
    for y in range(height):
        for x in range(width):
            red, green, blue = data[x, y]
            distance = abs(red - background[0]) + abs(green - background[1]) + abs(blue - background[2])
            if distance > threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x < 0 or max_y < 0:
        return (0, 0, width, height)

    return (
        max(0, min_x - padding),
        max(0, min_y - padding),
        min(width, max_x + padding),
        min(height, max_y + padding),
    )


def augment_query_image(
    input_path: Path,
    output_path: Path,
    crop_min: int,
    crop_max: int,
    zoom: float,
    rotate_left: float,
    seed: int | None = None,
) -> Path:
    if crop_min < 0 or crop_max < 0:
        raise RuntimeError("--crop-min and --crop-max must be >= 0.")
    if crop_min > crop_max:
        raise RuntimeError("--crop-min must be <= --crop-max.")
    if zoom <= 0:
        raise RuntimeError("--zoom must be greater than 0.")

    rng = random.Random(seed)
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    background = estimate_background_color(image)
    product_left, product_top, product_right, product_bottom = estimate_product_bbox(image, background)

    left_crop = min(rng.randint(crop_min, crop_max), max(0, product_left))
    top_crop = min(rng.randint(crop_min, crop_max), max(0, product_top))
    right_crop = min(rng.randint(crop_min, crop_max), max(0, width - product_right))
    bottom_crop = min(rng.randint(crop_min, crop_max), max(0, height - product_bottom))

    image = image.crop((left_crop, top_crop, width - right_crop, height - bottom_crop))
    image = image.resize((width, height), resampling_filter("LANCZOS"))

    zoomed_width = max(1, int(width * zoom))
    zoomed_height = max(1, int(height * zoom))
    image = image.resize((zoomed_width, zoomed_height), resampling_filter("LANCZOS"))
    image = ImageOps.fit(
        image,
        (width, height),
        method=resampling_filter("LANCZOS"),
        centering=(0.5, 0.5),
    )

    if rotate_left:
        image = image.rotate(
            rotate_left,
            resample=resampling_filter("BICUBIC"),
            expand=False,
            fillcolor=background,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    LOG.info(
        "Augmented query image: %s (crop L%s T%s R%s B%s, zoom %.2f, rotate-left %.1f)",
        output_path,
        left_crop,
        top_crop,
        right_crop,
        bottom_crop,
        zoom,
        rotate_left,
    )
    return output_path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gallery", help="Folder containing gallery images.")
    parser.add_argument("--metadata", help="Optional JSON metadata from task 5 hot_product_images.json or image_candidates.json.")
    parser.add_argument("--index", default=str(default_index_path()), help="Path to cached gallery index JSON.")
    parser.add_argument("--mode", choices=["auto", "clip", "classic"], default="auto", help="auto tries CLIP first, then falls back to classic visual features.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="Transformers CLIP model id.")
    parser.add_argument("--device", default="auto", help="CLIP device: auto, cpu, cuda.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan gallery subfolders.")
    parser.add_argument("--rebuild-index", action="store_true", help="Ignore cached index and rebuild all features.")
    parser.add_argument("--verbose", action="store_true")


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Task 6: find top-N images similar to one input image.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="Build or refresh a gallery similarity index.")
    add_common_args(build)

    query = sub.add_parser("search", help="Search a gallery for images similar to the input image.")
    add_common_args(query)
    query.add_argument("--input", required=True, help="Query image path.")
    query.add_argument("--top-n", type=int, default=10)
    query.add_argument("--output", default="", help="Output run folder. Defaults to task6_image_similarity/output/run_TIMESTAMP.")
    query.add_argument("--augment-query", action="store_true", help="Crop, zoom, and rotate the query image before searching.")
    query.add_argument("--crop-min", type=int, default=20, help="Minimum random safe crop in pixels for --augment-query.")
    query.add_argument("--crop-max", type=int, default=50, help="Maximum random safe crop in pixels for --augment-query.")
    query.add_argument("--zoom", type=float, default=1.10, help="Zoom factor for --augment-query.")
    query.add_argument("--rotate-left", type=float, default=15.0, help="Counter-clockwise rotation in degrees for --augment-query.")
    query.add_argument("--augment-seed", type=int, default=None, help="Optional random seed for reproducible query augmentation.")
    query.add_argument("--save-augmented-input", default="", help="Where to save the augmented query image. Defaults to OUTPUT/augmented_query.jpg.")
    query.add_argument("--include-input", action="store_true", help="Allow the original input image to appear in search results.")
    query.add_argument("--gemini-rerank", action="store_true", help="Use Gemini Vision to rerank the best local candidates.")
    query.add_argument("--gemini-top-k", type=int, default=30, help="Number of local candidates to send to Gemini before final top-N.")
    query.add_argument("--gemini-model", default=gemini_env("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash"), help="Gemini model for --gemini-rerank.")
    query.add_argument("--gemini-backend", choices=["auto", "enterprise", "api-key"], default="auto", help="Gemini auth backend.")
    query.add_argument("--gemini-batch-size", type=int, default=5, help="Images per Gemini rerank call, max 8.")
    return parser.parse_args()


def prepare_embedder(args: argparse.Namespace):
    try:
        embedder = create_embedder(args.mode, args.clip_model, args.device)
    except Exception as exc:
        raise RuntimeError(f"Cannot start CLIP backend: {exc}") from exc
    if embedder:
        LOG.info("Using embedding backend: %s", embedder.name)
    else:
        LOG.warning("Using classic visual fallback: color + edge + perceptual hash + aspect ratio.")
    return embedder


def ensure_index(args: argparse.Namespace, embedder) -> dict:
    index_path = Path(args.index)
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    if args.gallery:
        return build_index(
            gallery=Path(args.gallery),
            index_path=index_path,
            metadata_path=metadata_path,
            recursive=not args.no_recursive,
            embedder=embedder,
            rebuild=args.rebuild_index,
        )
    payload = load_index(index_path)
    if not payload:
        raise RuntimeError("--gallery is required when the index does not exist.")
    return payload


def run_build_index(args: argparse.Namespace) -> int:
    if not args.gallery:
        raise RuntimeError("--gallery is required for build-index.")
    embedder = prepare_embedder(args)
    payload = ensure_index(args, embedder)
    LOG.info("Indexed %s images -> %s", payload.get("image_count"), Path(args.index))
    return 0


def run_search(args: argparse.Namespace) -> int:
    embedder = prepare_embedder(args)
    index_payload = ensure_index(args, embedder)
    index_has_embeddings = bool(index_payload.get("embedding_model"))
    query_embedder = embedder if index_has_embeddings else None
    output_dir = Path(args.output).resolve() if args.output else default_output_path()
    original_query_path = Path(args.input)
    query_path = original_query_path

    if args.augment_query:
        augmented_path = Path(args.save_augmented_input).resolve() if args.save_augmented_input else output_dir / "augmented_query.jpg"
        query_path = augment_query_image(
            input_path=query_path,
            output_path=augmented_path,
            crop_min=args.crop_min,
            crop_max=args.crop_max,
            zoom=args.zoom,
            rotate_left=args.rotate_left,
            seed=args.augment_seed,
        )

    query = query_features(query_path, query_embedder)
    exclude_paths = [] if args.include_input else [original_query_path, query_path]
    candidate_count = max(args.top_n, args.gemini_top_k) if args.gemini_rerank else args.top_n
    results = search(query=query, index_payload=index_payload, top_n=candidate_count, exclude_paths=exclude_paths)

    if args.gemini_rerank:
        results = rerank_with_gemini(
            query_path=query_path,
            results=results,
            model=args.gemini_model,
            backend=args.gemini_backend,
            batch_size=args.gemini_batch_size,
            raw_output_dir=output_dir / "gemini_raw",
        )[: args.top_n]
        for rank, item in enumerate(results, start=1):
            item.rank = rank

    export_results(output_dir, query=query, results=results, index_payload=index_payload)
    LOG.info("Report: %s", output_dir / "similarity_report.html")
    LOG.info("CSV:    %s", output_dir / "similarity_results.csv")
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.command == "build-index":
        return run_build_index(args)
    if args.command == "search":
        return run_search(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
