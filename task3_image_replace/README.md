# Task 3 - Rug Artwork Replacement

## Tool Trong Thư Mục

Task này chứa tool thay artwork/pattern vào ảnh rug reference và benchmark kết quả bằng Gemini image models.

File chính:

- `benchmark_rug_artwork_replacement.py`
- `dataset/`: ảnh artwork nguồn và rug reference mẫu.
- `runs_task3_4k/`: output benchmark đã sinh.
- `Task3_Rug_Artwork_Replacement_Benchmark_Report.docx`: báo cáo thủ công/tổng kết.

## Tác Dụng

So sánh các model/mode khi cần chuyển artwork từ ảnh input sang nhiều ảnh rug reference, đồng thời ghi lại chi phí, latency và file review.

## Logic Chính

1. Nhận một artwork source và nhiều rug reference.
2. Tạo prompt giữ hình dáng/chất liệu/framing rug nhưng thay artwork.
3. Gửi request tới Gemini image model theo matrix model/mode/target.
4. Lưu output, thumbnail, manifest, CSV.
5. Tạo HTML report và manual review template để chấm chất lượng.

## Cấu Hình

`.env` mẫu:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_ENTERPRISE=True
USD_TO_VND=26000
```

Yêu cầu:

- Python virtual environment đã cài `google-genai`, `pillow`, `numpy`, `pandas`.
- Google ADC đã login.

## Cách Dùng

```powershell
cd D:\CODE\Python\Tool_Pinterest\task3_image_replace
python benchmark_rug_artwork_replacement.py --help
python benchmark_rug_artwork_replacement.py
```

Nếu không truyền file/folder qua CLI, script có thể mở picker bằng `tkinter` khi môi trường hỗ trợ GUI.

## Output

Output mặc định nằm trong `runs_task3_*`, gồm:

- `outputs/`
- `thumbnails/`
- `results.csv`
- `summary.csv`
- `manifest.json`
- `report.html`
- `manual_review_template.csv`

## Git Ignore Của Task

`.gitignore` bỏ qua `.env`, `runs_task3*`, cache Python và file tạm. Dataset mẫu vẫn có thể được giữ lại nếu cần tái lập benchmark.

