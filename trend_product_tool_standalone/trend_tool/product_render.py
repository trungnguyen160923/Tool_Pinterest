from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .config import ProductTarget


@dataclass(frozen=True)
class ProductRenderRecord:
    source_path: Path
    print_path: Path
    product_path: Path
    mask_path: Path
    product_name: str
    shape: str
    width: int
    height: int
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def render_product_from_print(
    *,
    source_path: Path,
    print_path: Path,
    product_path: Path,
    mask_path: Path,
    target: ProductTarget,
    max_long_edge: int = 1800,
) -> ProductRenderRecord:
    product_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(print_path) as opened:
        artwork = ImageOps.exif_transpose(opened).convert("RGBA")
    if target.name == "blanket":
        product, mask = render_blanket_product(artwork, target, max_long_edge)
        notes = "print artwork rendered as a soft blanket product asset"
    else:
        product, mask = render_rug_product(artwork, target, max_long_edge)
        notes = f"print artwork rendered as a {target.rug_shape} rug product asset"
    product.save(product_path)
    mask.save(mask_path)
    return ProductRenderRecord(
        source_path=source_path,
        print_path=print_path,
        product_path=product_path,
        mask_path=mask_path,
        product_name=target.name,
        shape=target.rug_shape if target.name == "rug" else "rectangle",
        width=product.width,
        height=product.height,
        status="ok",
        notes=notes,
    )


def render_rug_product(
    artwork: Image.Image,
    target: ProductTarget,
    max_long_edge: int,
) -> tuple[Image.Image, Image.Image]:
    shape = normalized_rug_shape(target.rug_shape)
    product_size = rug_canvas_size(shape, target, max_long_edge)
    body_box = inset_box(product_size, 0.045, 0.03)
    radius = max(18, min(product_size) // 28)

    canvas = Image.new("RGBA", product_size, (255, 255, 255, 0))
    mask = Image.new("L", product_size, 0)
    draw_mask = ImageDraw.Draw(mask)
    if shape in {"round", "oval"}:
        draw_mask.ellipse(body_box, fill=255)
    else:
        draw_mask.rounded_rectangle(body_box, radius=radius, fill=255)

    art = ImageOps.fit(artwork.convert("RGBA"), (body_box[2] - body_box[0], body_box[3] - body_box[1]), Image.Resampling.LANCZOS)
    art = add_textile_surface(art, strength=0.13)
    canvas.alpha_composite(art, (body_box[0], body_box[1]))
    canvas.putalpha(mask)

    canvas = add_edge_binding(canvas, body_box, radius, opacity=58, oval=shape in {"round", "oval"})
    if shape in {"rectangle", "runner"}:
        canvas = add_rug_fringe(canvas, body_box)
    mask = canvas.getchannel("A")
    return canvas, mask


def render_blanket_product(
    artwork: Image.Image,
    target: ProductTarget,
    max_long_edge: int,
) -> tuple[Image.Image, Image.Image]:
    product_size = product_canvas_size(target, max_long_edge)
    body_box = inset_box(product_size, 0.055, 0.045)
    radius = max(24, min(product_size) // 22)

    canvas = Image.new("RGBA", product_size, (255, 255, 255, 0))
    mask = Image.new("L", product_size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle(body_box, radius=radius, fill=255)

    art = ImageOps.fit(artwork.convert("RGBA"), (body_box[2] - body_box[0], body_box[3] - body_box[1]), Image.Resampling.LANCZOS)
    art = add_textile_surface(art, strength=0.09)
    art = add_blanket_folds(art)
    canvas.alpha_composite(art, (body_box[0], body_box[1]))
    canvas.putalpha(mask.filter(ImageFilter.GaussianBlur(0.6)))
    canvas = add_edge_binding(canvas, body_box, radius, opacity=46)
    mask = canvas.getchannel("A")
    return canvas, mask


def product_canvas_size(target: ProductTarget, max_long_edge: int) -> tuple[int, int]:
    ratio = max(0.1, target.width_px / max(1, target.height_px))
    long_edge = max(900, int(max_long_edge))
    if ratio >= 1:
        width = long_edge
        height = max(1, round(long_edge / ratio))
    else:
        height = long_edge
        width = max(1, round(long_edge * ratio))
    return width, height


def normalized_rug_shape(value: str) -> str:
    return value.strip().lower() if value.strip().lower() in {"rectangle", "square", "round", "oval", "runner"} else "rectangle"


def rug_canvas_size(shape: str, target: ProductTarget, max_long_edge: int) -> tuple[int, int]:
    long_edge = max(900, int(max_long_edge))
    if shape in {"round", "square"}:
        return long_edge, long_edge
    if shape == "oval":
        return long_edge, max(1, round(long_edge * 0.68))
    if shape == "runner":
        return max(1, round(long_edge * 0.42)), long_edge
    return product_canvas_size(target, long_edge)


def inset_box(size: tuple[int, int], x_ratio: float, y_ratio: float) -> tuple[int, int, int, int]:
    width, height = size
    x = max(2, round(width * x_ratio))
    y = max(2, round(height * y_ratio))
    return x, y, width - x, height - y


def add_textile_surface(image: Image.Image, *, strength: float) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_alpha = max(5, round(255 * strength))
    for y in range(0, image.height, 12):
        draw.line((0, y, image.width, y), fill=(255, 255, 255, line_alpha), width=1)
    for x in range(0, image.width, 14):
        draw.line((x, 0, x, image.height), fill=(0, 0, 0, max(3, line_alpha // 3)), width=1)
    return Image.alpha_composite(image, overlay)


def add_blanket_folds(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    shade = Image.new("L", image.size, 128)
    draw = ImageDraw.Draw(shade)
    for x in range(max(24, image.width // 7), image.width, max(36, image.width // 5)):
        draw.line((x, 0, x - image.width // 18, image.height), fill=98, width=max(10, image.width // 55))
        draw.line((x + image.width // 34, 0, x - image.width // 30, image.height), fill=164, width=max(8, image.width // 70))
    shade = shade.filter(ImageFilter.GaussianBlur(radius=max(8, image.width // 70)))
    darker = ImageEnhance.Brightness(rgb).enhance(0.88)
    lighter = ImageEnhance.Brightness(rgb).enhance(1.08)
    folded = Image.composite(lighter, darker, shade)
    folded.putalpha(image.getchannel("A"))
    return folded


def add_edge_binding(
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    *,
    opacity: int,
    oval: bool = False,
) -> Image.Image:
    border = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(border)
    draw_method = draw.ellipse if oval else draw.rounded_rectangle
    if oval:
        draw_method(box, outline=(255, 255, 255, opacity), width=max(3, min(image.size) // 70))
        draw_method(box, outline=(0, 0, 0, max(12, opacity // 3)), width=max(1, min(image.size) // 160))
    else:
        draw_method(box, radius=radius, outline=(255, 255, 255, opacity), width=max(3, min(image.size) // 70))
        draw_method(box, radius=radius, outline=(0, 0, 0, max(12, opacity // 3)), width=max(1, min(image.size) // 160))
    return Image.alpha_composite(image, border)


def add_rug_fringe(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    fringe = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fringe)
    left, top, right, bottom = box
    length = max(10, image.height // 45)
    spacing = max(8, image.width // 90)
    for x in range(left + spacing, right - spacing, spacing):
        draw.line((x, top, x, max(0, top - length)), fill=(238, 232, 220, 210), width=2)
        draw.line((x, bottom, x, min(image.height - 1, bottom + length)), fill=(238, 232, 220, 210), width=2)
    combined = Image.alpha_composite(fringe, image)
    alpha = ImageChops.lighter(fringe.getchannel("A"), image.getchannel("A"))
    combined.putalpha(alpha)
    return combined
