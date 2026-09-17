from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .config import ProductTarget


def enhance_image(source: Path, destination: Path, min_long_edge: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        scale = max(1, min_long_edge / max(image.size))
        if scale > 1:
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.6, percent=120, threshold=3))
        image.save(destination)
    return destination


def fit_to_target(source: Path, destination: Path, target: ProductTarget, mode: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        if mode == "none":
            fitted = image.resize((target.width_px, target.height_px), Image.Resampling.LANCZOS)
        elif mode == "contain":
            fitted = contain_on_canvas(image, (target.width_px, target.height_px))
        else:
            fitted = ImageOps.fit(image, (target.width_px, target.height_px), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        fitted.save(destination, dpi=(target.dpi, target.dpi))
    return destination


def contain_on_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.alpha_composite(contained, offset)
    return canvas


def remove_near_white_background(source: Path, destination: Path, threshold: int = 245) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        pixels = rgba.load()
        for y in range(rgba.height):
            for x in range(rgba.width):
                red, green, blue, alpha = pixels[x, y]
                if red >= threshold and green >= threshold and blue >= threshold:
                    pixels[x, y] = (red, green, blue, 0)
                else:
                    pixels[x, y] = (red, green, blue, alpha)
        rgba.save(destination)
    return destination


def export_cmyk_jpg(source: Path, destination: Path, dpi: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        flattened = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            flattened.paste(image, mask=image.getchannel("A"))
        else:
            flattened.paste(image.convert("RGB"))
        cmyk = flattened.convert("CMYK")
        cmyk.save(destination, quality=95, dpi=(dpi, dpi))
    return destination

