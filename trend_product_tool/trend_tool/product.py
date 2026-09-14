from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .config import ProductTarget


MOCKUP_BACKGROUNDS = (
    (238, 233, 226),
    (214, 222, 219),
    (226, 217, 206),
    (231, 228, 218),
)


def make_product_mockups(print_file: Path, destination_dir: Path, target: ProductTarget, count: int) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    with Image.open(print_file) as image:
        design = ImageOps.exif_transpose(image).convert("RGBA")
        for index, bg_color in enumerate(MOCKUP_BACKGROUNDS[:count], start=1):
            mockup = render_mockup(design, bg_color, target)
            out_path = destination_dir / f"{target.name}_mockup_{index:02d}.jpg"
            mockup.convert("RGB").save(out_path, quality=92)
            output.append(out_path)
    return output


def render_mockup(design: Image.Image, bg_color: tuple[int, int, int], target: ProductTarget) -> Image.Image:
    canvas_size = (1600, 1200)
    canvas = Image.new("RGB", canvas_size, bg_color)
    draw = ImageDraw.Draw(canvas)
    if target.name == "blanket":
        draw.rectangle((0, 0, 1600, 420), fill=tuple(min(255, c + 14) for c in bg_color))
        draw.rounded_rectangle((285, 360, 1315, 980), radius=26, fill=tuple(max(0, c - 20) for c in bg_color))
        draw.rounded_rectangle((360, 300, 1240, 470), radius=22, fill=tuple(min(255, c + 26) for c in bg_color))
        product_anchor_y = 420
        max_w = 820
        max_h = 520
    elif target.name == "custom":
        draw.rectangle((0, 780, 1600, 1200), fill=tuple(max(0, c - 18) for c in bg_color))
        draw.ellipse((465, 790, 1135, 1045), fill=tuple(max(0, c - 32) for c in bg_color))
        product_anchor_y = 250
        max_w = 760
        max_h = 680
    else:
        draw.rectangle((0, 700, 1600, 1200), fill=tuple(max(0, c - 18) for c in bg_color))
        for x in range(-120, 1720, 180):
            draw.line((x, 700, x + 300, 1200), fill=tuple(max(0, c - 32) for c in bg_color), width=2)
        product_anchor_y = 360
        max_w = 820
        max_h = 720

    product = ImageOps.contain(design, (max_w, max_h), Image.Resampling.LANCZOS)
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    alpha = product.getchannel("A") if product.mode == "RGBA" else Image.new("L", product.size, 255)
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=22)))
    shadow_offset = ((canvas_size[0] - product.width) // 2 + 18, product_anchor_y + 22)
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(shadow, shadow_offset)
    canvas_rgba.alpha_composite(product, ((canvas_size[0] - product.width) // 2, product_anchor_y))
    return canvas_rgba.convert("RGB")
