from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .crawler import CandidateImage
from .dedupe import DedupeDecision


@dataclass(frozen=True)
class Task6SimilarityConfig:
    gallery: Path
    index_path: Path | None = None
    metadata_path: Path | None = None
    threshold: float = 92.0
    recursive: bool = True
    rebuild_index: bool = False
    mode: str = "auto"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"


def reject_matches_from_task6(
    candidates: list[CandidateImage],
    config: Task6SimilarityConfig | None,
) -> tuple[list[CandidateImage], list[DedupeDecision]]:
    if config is None:
        return candidates, []
    modules = load_task6_modules()
    build_index = modules["build_index"]
    query_features = modules["query_features"]
    search = modules["search"]
    create_embedder = modules["create_embedder"]

    mode = (config.mode or "auto").strip().lower()
    embedder = create_embedder(mode, config.clip_model, config.device)

    index_path = config.index_path or config.gallery / ".task6_gallery_index.json"
    index_payload = build_index(
        gallery=config.gallery,
        index_path=index_path,
        metadata_path=config.metadata_path,
        recursive=config.recursive,
        embedder=embedder,
        rebuild=config.rebuild_index,
    )

    accepted: list[CandidateImage] = []
    decisions: list[DedupeDecision] = []
    for candidate in candidates:
        query = query_features(candidate.path, embedder)
        matches = search(query=query, index_payload=index_payload, top_n=1, exclude_paths=[candidate.path])
        best = matches[0] if matches else None
        if best and best.score >= config.threshold:
            decisions.append(
                DedupeDecision(
                    candidate=candidate,
                    kept=False,
                    reason=f"task6_match_score_{best.score:.1f}",
                    distance=None,
                    duplicate_of=Path(best.path),
                )
            )
            continue
        accepted.append(candidate)
    return accepted, decisions


def load_task6_modules() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    task6_root = repo_root / "task6_image_similarity"
    if not task6_root.exists():
        raise RuntimeError(f"task6_image_similarity not found at {task6_root}")
    task6_root_text = str(task6_root)
    if task6_root_text not in sys.path:
        sys.path.insert(0, task6_root_text)

    from image_similarity.indexer import build_index
    from image_similarity.searcher import query_features, search
    from image_similarity.embeddings import create_embedder

    return {
        "build_index": build_index,
        "query_features": query_features,
        "search": search,
        "create_embedder": create_embedder,
    }
