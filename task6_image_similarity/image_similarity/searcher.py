from __future__ import annotations

from pathlib import Path

from .embeddings import ClipEmbedder
from .features import (
    aspect_similarity,
    cosine_similarity,
    extract_query_features,
    hash_similarity,
    histogram_intersection,
)
from .models import ImageRecord, QueryFeatures, SimilarityResult


def query_features(path: Path, embedder: ClipEmbedder | None = None) -> QueryFeatures:
    features = extract_query_features(path)
    if embedder:
        features.embedding = embedder.embed_image(path)
    return features


def score_record(query: QueryFeatures, record: ImageRecord) -> tuple[float, dict[str, float | None]]:
    semantic = cosine_similarity(query.embedding, record.embedding)
    color = histogram_intersection(query.color_hist, record.color_hist)
    edge = histogram_intersection(query.edge_hist, record.edge_hist)
    hash_score = hash_similarity(query.dhash, record.dhash)
    aspect = aspect_similarity(query.aspect_ratio, record.aspect_ratio)

    if semantic is None:
        score = color * 0.45 + edge * 0.25 + hash_score * 0.20 + aspect * 0.10
    else:
        score = semantic * 0.75 + color * 0.10 + edge * 0.07 + hash_score * 0.05 + aspect * 0.03

    return score, {
        "semantic_score": semantic,
        "color_score": color,
        "edge_score": edge,
        "hash_score": hash_score,
        "aspect_score": aspect,
    }


def search(
    *,
    query: QueryFeatures,
    index_payload: dict,
    top_n: int = 10,
) -> list[SimilarityResult]:
    records = [
        item if isinstance(item, ImageRecord) else ImageRecord(**item)
        for item in index_payload.get("records", [])
    ]
    scored: list[SimilarityResult] = []
    for record in records:
        score, parts = score_record(query, record)
        scored.append(
            SimilarityResult(
                rank=0,
                image_id=record.image_id,
                path=record.path,
                filename=record.filename,
                width=record.width,
                height=record.height,
                score=round(score * 100.0, 2),
                semantic_score=(
                    round(float(parts["semantic_score"]) * 100.0, 2)
                    if parts["semantic_score"] is not None
                    else None
                ),
                color_score=round(float(parts["color_score"]) * 100.0, 2),
                edge_score=round(float(parts["edge_score"]) * 100.0, 2),
                hash_score=round(float(parts["hash_score"]) * 100.0, 2),
                aspect_score=round(float(parts["aspect_score"]) * 100.0, 2),
                metadata=record.metadata,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    selected = scored[: max(1, top_n)]
    for rank, item in enumerate(selected, start=1):
        item.rank = rank
    return selected
