# Task 5 - Pinterest Hot Trend

## Tool Trong Thư Mục

Task này là workbench để tìm trend Pinterest, crawl ảnh theo trend, lọc ảnh sản phẩm và xuất report review.

File/folder chính:

- `pinterest_trend_finder.py`: wrapper CLI chạy trend finder.
- `hot_image_crawler.py`: wrapper CLI chạy image crawler.
- `pinterest_browser_login.py`: mở browser profile để login Pinterest.
- `web_app.py`: web UI local bằng `http.server`.
- `trend_finder/`: logic lấy trend và semantic analysis.
- `image_crawler/`: logic search/download/dedupe/vision filter/rank ảnh.
- `shared/`: dataclass, cache, utils, product policy.
- `web_ui/`: CSS/JS của workbench.

## Tác Dụng

1. Tìm xu hướng liên quan tới một niche.
2. Sinh `trend_package.json`.
3. Crawl ảnh từ Pinterest/Bing theo query.
4. Download ảnh, loại duplicate bằng perceptual hash.
5. Lọc ảnh bằng Gemini vision theo product policy.
6. Rank ảnh hot product.
7. Xuất JSON/CSV/HTML report.

## Logic Chính

Trend finder:

1. Gọi Pinterest Trends endpoints khi có token/API hợp lệ.
2. Chuẩn hóa candidates từ keyword/topic/shopping/editorial.
3. Dùng semantic analyzer để đánh giá fit với niche.
4. Sinh query crawl theo trend.
5. Xuất `trend_package.json`, CSV và report.

Image crawler:

1. Đọc `trend_package.json`.
2. Search ảnh theo provider.
3. Download ảnh về `downloaded_images`.
4. Dedupe bằng `dhash`.
5. Vision filter kiểm tra product presence, role, visibility, trend relevance.
6. Rank ảnh bằng score tổng hợp.
7. Xuất `hot_product_images.*`, `rejected_images.json`, `crawl_manifest.json`.

## Cấu Hình

`.env` mẫu:

```env
PINTEREST_ACCESS_TOKEN=...
PINTEREST_LOCALE=en-US
PINTEREST_TIMEOUT=30
GEMINI_ANALYSIS_MODEL=gemini-2.5-flash
GEMINI_VISION_MODEL=gemini-2.5-flash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_ENTERPRISE=True
```

OAuth/token/browser profile không nên commit:

- `.env`
- `.pinterest_oauth_tokens.json`
- `.pinterest_browser_profile/`

## Cách Dùng CLI

Tìm trend:

```powershell
cd D:\CODE\Python\Tool_Pinterest\task5_hottrend
python pinterest_trend_finder.py --niche blanket --region US --output blanket_trend_output --max-trends 20 --verbose
```

Crawl ảnh:

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

Web UI:

```powershell
python web_app.py
```

Mở:

```text
http://127.0.0.1:8787
```

## Output

Trend output:

- `trend_package.json`
- `trend_package.csv`
- `trends_raw.json`
- `trend_report.html`

Crawl output:

- `downloaded_images/`
- `raw_results.json`
- `image_candidates.json`
- `product_vision_analysis.json`
- `hot_product_images.json`
- `hot_product_images.csv`
- `rejected_images.json`
- `hot_product_images_report.html`
- `crawl_manifest.json`

## Git Ignore Của Task

`.gitignore` bỏ qua secret, browser profile, temporary probe folders, trend/crawl output, cache Python và generated reports.

