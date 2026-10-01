# Tool_Pinterest — Trend Product Tool (Standalone)

Hệ thống tự động hóa toàn diện từ khám phá xu hướng Pinterest, thẩm định khả năng in ấn (Printability & Quality review), thiết kế mẫu vải/sản phẩm (Rug/Blanket), xuất file master in CMYK 300 DPI và render mockup phối cảnh (Direct AI, Template AI, và Blender 3D Mesh UV).

Dự án hiện đã được tái cấu trúc thành một tool độc lập duy nhất (100% self-contained) nằm trong thư mục [`trend_product_tool_standalone/`](trend_product_tool_standalone/).

---

## Cấu Trúc Dự Án

```
Tool_Pinterest/
├── trend_product_tool_standalone/    # Bộ công cụ chính (self-contained)
│   ├── api.py                        # FastAPI backend (REST API)
│   ├── app.py                        # Streamlit UI dashboard
│   ├── run.py                        # Runner chạy đồng thời UI & API
│   ├── requirements.txt              # Danh sách thư viện Python
│   ├── docs/                         # Tài liệu và báo cáo benchmark lưu trữ
│   ├── pinterest/                    # Module crawl xu hướng & ảnh từ Pinterest
│   ├── third_party/                  # Blender portable bundled runtime
│   └── trend_tool/                   # Pipeline core, AI rendering & color management
├── .gitignore
└── README.md
```

---

## Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường

```powershell
cd trend_product_tool_standalone
pip install -r requirements.txt
playwright install chromium
Copy-Item .env.example .env
```

Cấu hình các API key cần thiết trong `.env` (`GEMINI_API_KEY`, Pinterest credentials,...).

### 2. Chạy ứng dụng

#### Chạy đồng thời cả UI (Streamlit) và API (FastAPI)
```powershell
python run.py
```
- UI chạy tại: `http://localhost:8502`
- API docs (Swagger UI) tại: `http://localhost:8000/docs`

#### Chạy riêng UI
```powershell
streamlit run app.py --server.port 8501
```

#### Chạy riêng REST API
```powershell
python api.py
```

### 3. Kiểm tra & Chạy Unit Tests

```powershell
python -m compileall -x "third_party" trend_product_tool_standalone
python -m unittest test_pattern_review_pipeline.py
```
