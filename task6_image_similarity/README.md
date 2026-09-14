# Task 6 - Image Similarity Search

Tool tìm top-N ảnh tương đồng với một ảnh input trong một kho ảnh.

## Tool Trong Thư Mục

File/folder chính:

- `image_similarity_tool.py`: CLI entrypoint.
- `image_similarity/`: package lõi tính feature, embedding, index, search, report.
- `requirements.txt`: dependency cho CLIP semantic backend.
- `index/`: cache index sinh ra khi chạy.
- `output/`: report/CSV/JSON của mỗi lần search.

## Tác Dụng

Đánh giá độ tương đồng của một ảnh input với một kho ảnh, ví dụ kho có 100 hoặc 200 ảnh, rồi hiển thị top-N ảnh giống nhất.

Best-practice flow:

1. Build/cache image index.
2. Use image embeddings as the primary similarity signal when CLIP is available.
3. Rerank with visual signals: color, edge/texture, perceptual hash, aspect ratio.
4. Export JSON, CSV, and an HTML report for human review.

## Logic Chính

1. Scan folder gallery và lọc các ảnh hỗ trợ: JPG, PNG, WEBP, BMP, TIFF.
2. Tính classic features: `dhash`, color histogram, edge/texture histogram, aspect ratio.
3. Nếu dùng `--mode clip` hoặc `--mode auto` và dependency đủ, tính CLIP image embedding.
4. Cache toàn bộ index vào `index/gallery_index.json`.
5. Với ảnh input, tính feature/embedding tương tự.
6. So sánh bằng cosine similarity + hybrid rerank.
7. Xuất top-N vào HTML/CSV/JSON.

## Quick Start

Classic fallback, no heavy model required:

```powershell
python task6_image_similarity\image_similarity_tool.py search `
  --input task5_hottrend\blanket_crawl_output\downloaded_images\001cee8a1151e8ac.jpg `
  --gallery task5_hottrend\blanket_crawl_output\downloaded_images `
  --metadata task5_hottrend\blanket_crawl_output\image_candidates.json `
  --top-n 10 `
  --mode classic `
  --rebuild-index
```

Augmented query test: crop safe random 20-50px, zoom 110%, rotate left 15 degrees before searching.

```powershell
python task6_image_similarity\image_similarity_tool.py search `
  --input task5_hottrend\blanket_crawl_output\downloaded_images\001cee8a1151e8ac.jpg `
  --gallery task5_hottrend\blanket_crawl_output\downloaded_images `
  --metadata task5_hottrend\blanket_crawl_output\image_candidates.json `
  --top-n 10 `
  --mode classic `
  --augment-query `
  --crop-min 20 `
  --crop-max 50 `
  --zoom 1.10 `
  --rotate-left 15 `
  --save-augmented-input .tmp\augmented_query.jpg `
  --rebuild-index
```

If the product touches the image border, safe crop is automatically reduced on that side to avoid cutting the product. By default, the original input image is excluded from results so the top-N list does not return the query itself. Use `--include-input` only when you intentionally want to see self-matches.
Gemini rerank mode: first find local candidates, then send only the best candidates to Gemini Vision for human-like visual reranking.

```powershell
python task6_image_similarity\image_similarity_tool.py search `
  --input task5_hottrend\blanket_crawl_output\downloaded_images\mau.jpg `
  --gallery task5_hottrend\blanket_crawl_output\downloaded_images `
  --metadata task5_hottrend\blanket_crawl_output\image_candidates.json `
  --top-n 10 `
  --mode classic `
  --augment-query `
  --gemini-rerank `
  --gemini-top-k 30 `
  --gemini-model gemini-2.5-flash `
  --rebuild-index
```

`--gemini-rerank` is optional because it uses paid API calls. `--gemini-top-k 30` means Gemini only reviews the best 30 local candidates, not the entire gallery.

Embedding-first mode:

```powershell
pip install -r task6_image_similarity\requirements.txt

python task6_image_similarity\image_similarity_tool.py search `
  --input path\to\query.jpg `
  --gallery path\to\gallery `
  --metadata task5_hottrend\blanket_crawl_output\image_candidates.json `
  --top-n 10 `
  --mode auto
```

`--mode auto` tries CLIP first and falls back to classic visual scoring if the semantic backend is not installed.

## Cấu Hình

Không cần `.env` cho mode classic hoặc CLIP local. Lần đầu dùng CLIP sẽ tải model Hugging Face:

```text
openai/clip-vit-base-patch32
```

Có thể đổi model:

```powershell
python task6_image_similarity\image_similarity_tool.py search `
  --input path\to\query.jpg `
  --gallery path\to\gallery `
  --mode clip `
  --clip-model openai/clip-vit-large-patch14
```

Nếu đã build index CLIP, các lần query sau không cần `--rebuild-index`.

## Outputs

Each search run writes:

- `similarity_report.html`
- `similarity_results.json`
- `similarity_results.csv`

Default output folder:

```text
task6_image_similarity/output/run_YYYYMMDD_HHMMSS
```

Default index cache:

```text
task6_image_similarity/index/gallery_index.json
```

## Scoring

When CLIP embeddings are available:

```text
final_score =
  75% semantic similarity
+ 10% color similarity
+  7% edge/texture similarity
+  5% perceptual hash similarity
+  3% aspect/layout similarity
```

When CLIP is unavailable:

```text
final_score =
  55% color similarity
+ 30% edge/texture similarity
+ 12% perceptual hash similarity
+  3% aspect/layout similarity
```

Classic mode is useful for smoke tests and near-duplicate/image-vibe matching. CLIP mode is the recommended mode for Pinterest-style semantic similarity.

## Git Ignore Của Task

`.gitignore` bỏ qua index cache, output report, Python cache, model/runtime temp và environment local.
