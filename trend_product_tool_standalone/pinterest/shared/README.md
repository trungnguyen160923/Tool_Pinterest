# Task 5 Component - Shared

## Chứa Gì

Package shared chứa model dữ liệu, cache, utility và product policy dùng chung cho trend finder, crawler và web app.

File chính:

- `models.py`: dataclass contract như `TrendPackage`, `ImageCandidate`, `VisionResult`, `RankedImage`.
- `utils.py`: env loader, JSON/CSV writer, HTML page helper, text normalize, stable ID.
- `cache.py`: JSON cache.
- `product_policy.py`: policy phân loại sản phẩm theo niche/focus.

## Tác Dụng

Giữ contract giữa các stage ổn định để trend finder, crawler và UI cùng đọc/ghi một format.

## Logic

- `TrendPackage` là input contract của crawler.
- `ImageCandidate` là ảnh đã search/download trước vision.
- `VisionResult` là output phân tích ảnh.
- `RankedImage` là output cuối để report.
- Product policy quyết định loại sản phẩm nào được accept/reject theo niche.

## Cấu Hình

`utils.py` load `.env` từ:

1. `trend_product_tool_standalone/.env`
2. legacy task5 env nếu có

Không lưu secret trong package này.
