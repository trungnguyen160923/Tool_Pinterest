# Task 4 - Product Background Replacement

## Tool Trong Thư Mục

Task này chứa pipeline thay nền sản phẩm theo hướng product-aware và semantic placement.

File chính:

- `benchmark_product_background_replace_semantic_v4_1_best_practice.py`
- `dataset/main.webp`: ảnh sản phẩm mẫu.
- `dataset/background.txt`: prompt nền tùy chọn.
- `Bao_cao_Task4.docx`: báo cáo tổng kết.

## Tác Dụng

Tạo ảnh sản phẩm trên nền mới một cách tự động, giữ sản phẩm gốc và cố gắng đặt sản phẩm vào bối cảnh hợp lý về phối cảnh, mặt đỡ, ánh sáng và bóng tiếp xúc.

## Logic Chính

1. Phân tích sản phẩm bằng vision model để hiểu loại vật thể, orientation, support surface.
2. Segment sản phẩm bằng local pipeline như `rembg` khi có.
3. Tự sinh background brief nếu không có `background.txt`.
4. Gọi Gemini image model để sinh nền.
5. Phân tích nền để tìm vị trí đặt sản phẩm hợp lý.
6. Composite bằng Pillow/OpenCV theo kế hoạch semantic.
7. Harmonize ánh sáng/màu và thêm contact shadow.
8. Tùy chọn chạy final image-model integration pass.
9. Xuất final PNG/HTML/CSV/manifest.

## Cấu Hình

`.env` mẫu:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_ENTERPRISE=True
USD_TO_VND=26000
```

Dependency chính:

```powershell
pip install -U google-genai pillow pandas python-dotenv numpy opencv-python-headless rembg onnxruntime
```

## Cách Dùng

```powershell
cd D:\CODE\Python\Tool_Pinterest\task4_background_replace
python benchmark_product_background_replace_semantic_v4_1_best_practice.py --dry-run
python benchmark_product_background_replace_semantic_v4_1_best_practice.py
```

Ví dụ truyền prompt nền:

```powershell
python benchmark_product_background_replace_semantic_v4_1_best_practice.py --background "A warm modern living room with natural daylight"
```

## Output

Các run nằm trong `runs_task4_*`, thường có:

- `inputs/`
- `intermediates/`
- `backgrounds/`
- `analysis/`
- `outputs/`
- `thumbnails/`
- `results.csv`
- `summary.csv`
- `report.html`
- `manifest.json`

## Git Ignore Của Task

`.gitignore` bỏ qua `.env`, `runs_task4*`, cache Python, intermediate/output lớn và file tạm.

