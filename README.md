# Tool_Pinterest

Workspace chứa các tool thử nghiệm cho workflow ảnh sản phẩm, Pinterest trend discovery, crawl ảnh hot trend, và tìm ảnh tương đồng.

## Cấu Trúc

| Folder | Vai trò |
| --- | --- |
| `task2_image_benchmark` | Benchmark upscale/enhance ảnh bằng Gemini image models. |
| `task3_image_replace` | Thay artwork trên rug/reference images và đo cost/latency/output. |
| `task4_background_replace` | Tách sản phẩm, sinh nền mới, đặt sản phẩm vào nền theo phân tích semantic. |
| `task5_hottrend` | Tìm trend Pinterest, crawl ảnh theo trend, lọc/rank ảnh sản phẩm. |
| `task6_image_similarity` | Tìm top-N ảnh giống một ảnh input trong kho ảnh bằng CLIP embedding + rerank. |

## Nguyên Tắc Chung

- Mỗi task có README riêng mô tả tool, logic, cấu hình, cách chạy.
- `.env`, OAuth token, browser profile, model cache, output crawl/report/run đều không nên commit.
- Các report HTML/CSV/JSON sinh ra trong `runs*`, `output`, `*_crawl_output`, `*_trend_output`, `.tmp*` là artifact chạy thử.
- Khi cần dùng Google/Gemini, cấu hình qua `.env` trong task tương ứng.
- Khi cần dùng task6 CLIP, cài dependency trong virtual environment rồi dùng `--mode clip` hoặc `--mode auto`.

## Git

Repo được khởi tạo ở root `Tool_Pinterest`. `.gitignore` tổng thể chặn secret/cache/output lớn; mỗi task cũng có `.gitignore` riêng để task tự bảo vệ artifact của nó.

## Quick Checks

```powershell
python -m compileall task6_image_similarity
python task6_image_similarity\image_similarity_tool.py --help
```

