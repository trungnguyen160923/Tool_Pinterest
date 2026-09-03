from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .embeddings import ClipEmbedder
from .features import extract_query_features
from .models import ImageRecord
from .utils import file_signature, iter_image_paths, read_json, stable_id, utc_now_iso, write_json


LOG = logging.getLogger("task6_image_similarity.indexer")


METADATA_FIELDS = (
    "image_id",
    "rank",
    "image_score",
    "trend_id",
    "trend",
    "query",
    "image_url",
    "pin_url",
    "pin_id",
    "source",
    "product_role",
    "product_visibility",
    "trend_relevance",
    "commercial_quality",
    "aesthetic",
    "detected_product",
    "main_subject",
    "target_product_type",
    "motifs",
    "reason",
)


def metadata_lookup(metadata_path: Path | None, gallery_root: Path) -> dict[str, dict[str, Any]]:
    if not metadata_path:
        return {}
    payload = read_json(metadata_path, [])
    if isinstance(payload, dict) and "images" in payload:
        payload = payload["images"]
    if not isinstance(payload, list):
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    bases = [gallery_root.resolve(), metadata_path.resolve().parent, metadata_path.resolve().parent.parent, Path.cwd().resolve()]
    for item in payload:
        if not isinstance(item, dict):
            continue
        local_path = str(item.get("local_path") or item.get("path") or "").strip()
        filtered = {key: item.get(key) for key in METADATA_FIELDS if key in item}
        if not filtered:
            filtered = dict(item)
        keys = set()
        if local_path:
            raw = Path(local_path)
            candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
            for candidate in candidates:
                keys.add(str(candidate.resolve()).lower())
            keys.add(raw.name.lower())
            keys.add(raw.stem.lower())
        image_id = str(item.get("image_id") or "").strip()
        if image_id:
            keys.add(image_id.lower())
        for key in keys:
            lookup.setdefault(key, filtered)
    return lookup


def metadata_for_path(path: Path, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resolved = str(path.resolve()).lower()
    return dict(lookup.get(resolved) or lookup.get(path.name.lower()) or lookup.get(path.stem.lower()) or {})


def load_index(index_path: Path) -> dict[str, Any] | None:
    payload = read_json(index_path, None)
    return payload if isinstance(payload, dict) else None


def build_index(
    *,
    gallery: Path,
    index_path: Path,
    metadata_path: Path | None = None,
    recursive: bool = True,
    embedder: ClipEmbedder | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    gallery = gallery.resolve()
    image_paths = iter_image_paths(gallery, recursive=recursive)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {gallery}")

    embedding_model = embedder.name if embedder else ""
    existing = None if rebuild else load_index(index_path)
    reusable: dict[str, dict[str, Any]] = {}
    if existing and existing.get("embedding_model", "") == embedding_model:
        for record in existing.get("records", []):
            if isinstance(record, dict):
                reusable[str(record.get("path", "")).lower()] = record

    lookup = metadata_lookup(metadata_path, gallery) if metadata_path else {}
    records: list[ImageRecord] = []
    reused = 0

    for index, path in enumerate(image_paths, start=1):
        size_bytes, mtime_ns = file_signature(path)
        old = reusable.get(str(path).lower())
        if (
            old
            and int(old.get("size_bytes") or -1) == size_bytes
            and int(old.get("mtime_ns") or -1) == mtime_ns
        ):
            old["metadata"] = metadata_for_path(path, lookup) or old.get("metadata") or {}
            records.append(ImageRecord(**old))
            reused += 1
            continue

        LOG.info("Indexing %s/%s: %s", index, len(image_paths), path.name)
        features = extract_query_features(path)
        embedding = embedder.embed_image(path) if embedder else None
        metadata = metadata_for_path(path, lookup)
        image_id = str(metadata.get("image_id") or stable_id(path.name, size_bytes, mtime_ns))
        records.append(
            ImageRecord(
                image_id=image_id,
                path=str(path),
                filename=path.name,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                width=features.width,
                height=features.height,
                aspect_ratio=features.aspect_ratio,
                dhash=features.dhash,
                color_hist=features.color_hist,
                edge_hist=features.edge_hist,
                embedding=embedding,
                metadata=metadata,
            )
        )

    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "gallery": str(gallery),
        "image_count": len(records),
        "embedding_model": embedding_model,
        "reused_records": reused,
        "records": records,
    }
    write_json(index_path, payload)
    return payload
