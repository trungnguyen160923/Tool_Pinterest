# Task 5 Component - Trend Finder

## Chứa Tool Gì

Package này chứa logic lấy và chuẩn hóa trend từ Pinterest, sau đó dùng semantic analysis để chọn trend phù hợp với niche.

File chính:

- `pinterest_trend_finder.py`: CLI/core orchestration.
- `pinterest_client.py`: client gọi Pinterest API.
- `semantic_analyzer.py`: Gemini semantic scoring; crawl query giữ nguyên keyword từ Pinterest Trends API.
- `models.py`: dataclass nội bộ cho trend candidates.

## Tác Dụng

Biến dữ liệu trend thô thành `trend_package.json` có contract rõ ràng cho crawler.

## Logic

1. Gọi các endpoint trend/keyword/topic/shopping/editorial.
2. Chuẩn hóa tên trend, source, rank, strength.
3. Dedupe trend theo normalized text.
4. Chấm visual inspiration fit; không yêu cầu trend phải thuộc đúng product category.
5. Giữ nguyên keyword Pinterest Trends API làm query crawl.
6. Xuất package/report.

## Cấu Hình

Đọc env qua package `pinterest.shared.utils`, ưu tiên `.env` trong `trend_product_tool_standalone`.

Biến quan trọng:

- `PINTEREST_ACCESS_TOKEN`
- `PINTEREST_TIMEOUT`
- `GEMINI_ANALYSIS_MODEL`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_GENAI_USE_ENTERPRISE`

## Cách Dùng

Thông thường chạy wrapper ở task root:

```powershell
python pinterest_trend_finder.py --niche blanket --region US --output blanket_trend_output --max-trends 20 --verbose
```
