from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .config import ProductTarget


@dataclass(frozen=True)
class DesignRecord:
    source_path: Path
    output_path: Path
    mode: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_print_design(source: Path, destination: Path, target: ProductTarget, mode: str) -> DesignRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = (mode or "product_design").strip().lower().replace("-", "_")
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
        source_size = image.size
        if mode == "direct":
            design = image
            notes = "source image passed through before print fitting"
        elif mode == "pattern_repeat":
            design = make_pattern_repeat(image, target)
            notes = "source converted into a repeat-style print design"
        else:
            design = make_product_design(image, target)
            notes = "source converted into a flat product-design candidate"
        design.save(destination)

    return DesignRecord(
        source_path=source,
        output_path=destination,
        mode=mode,
        source_width=source_size[0],
        source_height=source_size[1],
        output_width=design.width,
        output_height=design.height,
        notes=notes,
    )


def make_product_design(image: Image.Image, target: ProductTarget) -> Image.Image:
    """
    Local product-art isolation: crop the visual motif enough to behave like
    artwork, then place it on a clean print-ratio canvas.
    """
    has_alpha = has_meaningful_transparency(image)
    art = crop_alpha_border(image) if has_alpha else crop_low_value_border(image)
    art = normalize_artwork(art)
    canvas_size = intermediate_canvas_size(target)
    background = (255, 255, 255, 0) if has_alpha else background_from_image(art)
    canvas = Image.new("RGBA", canvas_size, background)
    fitted = ImageOps.contain(art, canvas_size, Image.Resampling.LANCZOS)
    offset = ((canvas.width - fitted.width) // 2, (canvas.height - fitted.height) // 2)
    canvas.alpha_composite(fitted, offset)
    return canvas


def make_pattern_repeat(image: Image.Image, target: ProductTarget) -> Image.Image:
    art = crop_low_value_border(normalize_artwork(image))
    canvas_size = intermediate_canvas_size(target)
    tile = ImageOps.contain(art, (max(320, canvas_size[0] // 2), max(320, canvas_size[1] // 2)), Image.Resampling.LANCZOS)
    tile = soften_tile_edges(tile)
    canvas = Image.new("RGBA", canvas_size, background_from_image(art))
    step_x = max(1, int(tile.width * 0.92))
    step_y = max(1, int(tile.height * 0.92))
    for y in range(-tile.height // 2, canvas.height + tile.height, step_y):
        for x in range(-tile.width // 2, canvas.width + tile.width, step_x):
            canvas.alpha_composite(tile, (x, y))
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 18))
    return Image.alpha_composite(canvas, overlay)


def intermediate_canvas_size(target: ProductTarget) -> tuple[int, int]:
    long_edge = min(max(target.width_px, target.height_px), 4096)
    ratio = target.width_px / target.height_px
    if target.width_px >= target.height_px:
        return long_edge, max(1, round(long_edge / ratio))
    return max(1, round(long_edge * ratio)), long_edge


def normalize_artwork(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.06)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=4))
    alpha = image.getchannel("A")
    rgb.putalpha(alpha)
    return rgb


def has_meaningful_transparency(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = image.getchannel("A")
    low, high = alpha.getextrema()
    return low < 245 and high > 16


def crop_alpha_border(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if not bbox:
        return rgba
    left, top, right, bottom = bbox
    pad_x = max(4, round((right - left) * 0.015))
    pad_y = max(4, round((bottom - top) * 0.015))
    return rgba.crop(
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(rgba.width, right + pad_x),
            min(rgba.height, bottom + pad_y),
        )
    )


def crop_low_value_border(image: Image.Image, threshold: int = 246) -> Image.Image:
    rgba = image.convert("RGBA")
    gray = rgba.convert("L")
    stat = ImageStat.Stat(gray)
    if stat.stddev[0] < 8:
        return rgba

    # Treat near-white/near-transparent margins as background before artwork placement.
    alpha = rgba.getchannel("A")
    bg = Image.new("L", rgba.size, 0)
    pixels = rgba.load()
    mask = bg.load()
    alpha_pixels = alpha.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, _ = pixels[x, y]
            if alpha_pixels[x, y] > 8 and not (r >= threshold and g >= threshold and b >= threshold):
                mask[x, y] = 255
    bbox = bg.getbbox()
    if not bbox:
        return rgba
    left, top, right, bottom = bbox
    pad_x = max(8, round((right - left) * 0.035))
    pad_y = max(8, round((bottom - top) * 0.035))
    box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(rgba.width, right + pad_x),
        min(rgba.height, bottom + pad_y),
    )
    if (box[2] - box[0]) * (box[3] - box[1]) < rgba.width * rgba.height * 0.08:
        return rgba
    return rgba.crop(box)


def soften_tile_edges(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    feather = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(feather)
    inset = max(8, min(rgba.size) // 24)
    if rgba.width > inset * 2 and rgba.height > inset * 2:
        draw.rectangle((inset, inset, rgba.width - inset, rgba.height - inset), fill=255)
    else:
        draw.rectangle((0, 0, rgba.width, rgba.height), fill=255)
    feather = feather.filter(ImageFilter.GaussianBlur(radius=max(2, inset // 2)))
    rgba.putalpha(ImageChops_multiply(alpha, feather))
    return rgba


def ImageChops_multiply(left: Image.Image, right: Image.Image) -> Image.Image:
    from PIL import ImageChops

    return ImageChops.multiply(left, right)


def background_from_image(image: Image.Image) -> tuple[int, int, int, int]:
    thumbnail = image.convert("RGBA")
    thumbnail.thumbnail((64, 64), Image.Resampling.LANCZOS)
    stat = ImageStat.Stat(thumbnail.convert("RGB"))
    rgb = tuple(max(0, min(255, int(round(value)))) for value in stat.median[:3])
    if max(rgb) < 35:
        rgb = (18, 18, 20)
    return rgb[0], rgb[1], rgb[2], 255
