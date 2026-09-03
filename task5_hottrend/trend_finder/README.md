# Task 5 Component - Trend Finder

## Chứa Tool Gì

Package này chứa logic lấy và chuẩn hóa trend từ Pinterest, sau đó dùng semantic analysis để chọn trend phù hợp với niche.

File chính:

- `pinterest_trend_finder.py`: CLI/core orchestration.
- `pinterest_client.py`: client gọi Pinterest API.
- `semantic_analyzer.py`: Gemini semantic scoring và query generation.
- `models.py`: dataclass nội bộ cho trend candidates.

## Tác Dụng

Biến dữ liệu trend thô thành `trend_package.json` có contract rõ ràng cho crawler.

## Logic

1. Gọi các endpoint trend/keyword/topic/shopping/editorial.
2. Chuẩn hóa tên trend, source, rank, strength.
3. Dedupe trend theo normalized text.
4. Chấm semantic fit với niche.
5. Sinh query crawl có priority.
6. Xuất package/report.

## Cấu Hình

Đọc env qua `task5_hottrend/shared/utils.py`, ưu tiên `.env` trong `task5_hottrend`.

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

