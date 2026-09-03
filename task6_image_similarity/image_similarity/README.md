# Task 6 Component - Image Similarity Package

## Chứa Tool Gì

Package này là lõi của task 6, dùng bởi `task6_image_similarity/image_similarity_tool.py`.

File chính:

- `embeddings.py`: optional CLIP backend bằng `torch` + `transformers`.
- `features.py`: classic visual features như color histogram, edge histogram, `dhash`.
- `indexer.py`: build/cache gallery index và ghép metadata từ task 5.
- `searcher.py`: tính similarity và sort top-N.
- `report.py`: xuất JSON/CSV/HTML report.
- `models.py`: dataclass cho record/query/result.
- `utils.py`: helper path, JSON, CSV, ID, timestamp.

## Tác Dụng

Tách logic similarity thành các module nhỏ để CLI, web UI hoặc pipeline khác có thể reuse.

## Logic

1. `indexer` scan gallery, tính feature/embedding, cache theo path/size/mtime.
2. `embeddings` load CLIP lazy; nếu thiếu dependency thì `auto` fallback.
3. `features` luôn chạy offline để có rerank visual.
4. `searcher` dùng cosine similarity cho embedding và hybrid score.
5. `report` render ảnh input + top results với điểm semantic/color/edge/hash/aspect.

## Scoring

CLIP mode:

```text
75% semantic + 10% color + 7% edge + 5% hash + 3% aspect
```

Classic fallback:

```text
45% color + 25% edge + 20% hash + 10% aspect
```

