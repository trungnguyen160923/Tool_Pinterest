#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from image_similarity.embeddings import create_embedder
from image_similarity.indexer import build_index, load_index
from image_similarity.report import export_results
from image_similarity.searcher import query_features, search
from image_similarity.utils import timestamp_slug


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
    parser = argparse.ArgumentParser(description="Task 6: find top-N images similar to one input image.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="Build or refresh a gallery similarity index.")
    add_common_args(build)

    query = sub.add_parser("search", help="Search a gallery for images similar to the input image.")
    add_common_args(query)
    query.add_argument("--input", required=True, help="Query image path.")
    query.add_argument("--top-n", type=int, default=10)
    query.add_argument("--output", default="", help="Output run folder. Defaults to task6_image_similarity/output/run_TIMESTAMP.")
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
    query = query_features(Path(args.input), query_embedder)
    results = search(query=query, index_payload=index_payload, top_n=args.top_n)

    output_dir = Path(args.output).resolve() if args.output else default_output_path()
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
