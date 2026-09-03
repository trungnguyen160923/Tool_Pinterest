from PIL import Image
from tkinter import Tk, filedialog

# Ẩn cửa sổ Tkinter chính
root = Tk()
root.withdraw()

# Mở hộp thoại chọn ảnh
image_path = filedialog.askopenfilename(
    title="Chọn ảnh cần kiểm tra",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
        ("All files", "*.*"),
    ],
)

if not image_path:
    print("Bạn chưa chọn ảnh.")
    raise SystemExit

with Image.open(image_path) as img:
    width, height = img.size

long_edge = max(width, height)

if long_edge >= 4096:
    level = "4K"
elif long_edge >= 2048:
    level = "2K"
elif long_edge >= 1024:
    level = "1K"
else:
    level = "Below 1K"

print(f"File: {image_path}")
print(f"Resolution: {width} x {height}")
print(f"Long edge: {long_edge}px")
print(f"Approx level: {level}")