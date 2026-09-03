# Task 2 - Image Benchmark

## Tool Trong Thư Mục

Task này chứa các script benchmark upscale/enhance ảnh bằng Gemini image models trên Google Cloud / Agent Platform.

Các file chính:

- `benchmark_vertex_best_practice_v2.py`: entrypoint chính, benchmark best-practice cho Vertex/Gemini image models.
- `benchmark_vertex_batch_practical.py`: biến thể batch/practical.
- `benchmark_vertex_flex_practical.py`: benchmark qua Flex PayGo.
- `check_img.py`, `test_vertex.py`: kiểm tra ảnh/API nhanh.
- `requirement.txt`: dependency cơ bản.

## Tác Dụng

Đánh giá chất lượng, latency và chi phí khi upscale/enhance ảnh lên các target như `1K`, `2K`, `4K`.

Output thường gồm:

- ảnh đầu ra theo model/target
- `results.csv`
- `summary.csv`
- `recommendations.csv`
- `report.html`
- `manifest.json`

## Logic Chính

1. Đọc ảnh từ `dataset` hoặc input CLI.
2. Chuẩn hóa kích thước/EXIF/aspect ratio.
3. Gửi ảnh tới các Gemini image models.
4. Lưu output theo model/target/run.
5. Tính metric tham khảo như SSIM, PSNR, sharpness, MAE khi có ground truth.
6. Tổng hợp cost/latency/token usage.
7. Xuất HTML report và CSV để review.

## Cấu Hình

Tạo `.env` trong thư mục task:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_ENTERPRISE=True
USD_TO_VND=26000
```

Cần Google ADC:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Cách Dùng

```powershell
cd D:\CODE\Python\Tool_Pinterest\task2_image_benchmark
pip install -r requirement.txt
python benchmark_vertex_best_practice_v2.py --mode practical
```

Chạy help để xem option cụ thể:

```powershell
python benchmark_vertex_best_practice_v2.py --help
```


## Biến Thể Benchmark

- `benchmark_vertex_best_practice_v2.py`: dùng mặc định cho benchmark Standard/PayGo; hỗ trợ `--mode practical` và `--mode scientific`.
- `benchmark_vertex_flex_practical.py`: dùng khi cần benchmark Flex PayGo và verify `traffic_type`.
- `benchmark_vertex_batch_practical.py`: dùng khi cần workflow batch/practical riêng.

File legacy `benchmark.py` đã được loại bỏ để tránh nhầm với bản best-practice hiện tại.

## Git Ignore Của Task

`.gitignore` trong task này bỏ qua `.env`, `.venv`, `runs*`, `tmp`, cache Python và output tạm.



