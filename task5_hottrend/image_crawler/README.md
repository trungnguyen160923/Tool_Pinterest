# Task 5 Component - Image Crawler

## Chứa Tool Gì

Package này chứa logic tìm, tải, dedupe, lọc và rank ảnh theo `trend_package.json`.

File chính:

- `hot_image_crawler.py`: CLI/core orchestration.
- `discovery.py`: provider search Pinterest/Bing.
- `downloader.py`: download ảnh và convert thành candidate.
- `dedupe.py`: perceptual hash duplicate filter.
- `vision_filter.py`: Gemini vision filter.
- `ranker.py`: scoring/ranking ảnh sản phẩm.
- `pinterest_browser_login.py`: login profile cho browser provider.

## Tác Dụng

Từ danh sách trend/query, tạo kho ảnh hot product đã lọc, có metadata và report để review.

## Logic

1. Đọc `trend_package.json`.
2. Search ảnh theo provider.
3. Download ảnh local.
4. Loại ảnh lỗi/trùng gần bằng `dhash`.
5. Dùng vision model kiểm tra product presence, role, visibility, trend relevance.
6. Áp product policy theo niche.
7. Rank ảnh và xuất top N.

## Cấu Hình

Provider:

- `auto`
- `pinterest-api`
- `pinterest-browser`
- `pinterest-web`
- `bing-images`

Biến env thường dùng:

- `PINTEREST_LOCALE`
- `PINTEREST_TIMEOUT`
- `GEMINI_VISION_MODEL`
- `GOOGLE_CLOUD_PROJECT`

## Cách Dùng

Chạy từ task root:

```powershell
python hot_image_crawler.py `
  --input blanket_trend_output\trend_package.json `
  --provider pinterest-browser `
  --output blanket_crawl_output `
  --max-downloads 200 `
  --top-images 100 `
  --vision-mode auto `
  --verbose
```

