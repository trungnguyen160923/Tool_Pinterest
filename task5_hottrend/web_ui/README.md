# Task 5 Component - Web UI

## Chứa Tool Gì

Folder này chứa frontend tĩnh cho `task5_hottrend/web_app.py`.

File chính:

- `app.js`: gọi API local, render metrics/tabs/gallery/table/log.
- `app.css`: layout workbench.
- `tokens.css`: design tokens.

## Tác Dụng

Cung cấp giao diện local để chạy trend finder, crawler, Pinterest browser login và review output.

## Logic

1. Browser gọi `/api/data` định kỳ để refresh status/output.
2. Button run gọi `/api/run/trends`, `/api/run/crawler`, `/api/run/browser-login`.
3. Data JSON được render thành các tab: accepted, trends, queries, rejected, raw URLs.
4. Ảnh local được serve qua endpoint `/file?path=...`.

## Cách Dùng

Chạy từ task root:

```powershell
python web_app.py
```

Mở:

```text
http://127.0.0.1:8787
```

