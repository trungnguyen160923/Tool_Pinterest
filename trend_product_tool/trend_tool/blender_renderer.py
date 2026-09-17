from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from .config import ProductTarget
from .template_mockup import TemplatePose


ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class BlenderMockupRecord:
    print_path: Path
    template_path: Path | None
    mask_path: Path | None
    mockup_path: Path | None
    model: str
    pose: str
    status: str
    notes: str
    metrics: dict[str, object]
    render_mode: str = "blender_3d"
    variant: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_blender_mockup(
    print_path: Path,
    output_dir: Path,
    target: ProductTarget,
    *,
    pose: TemplatePose,
    variant: int = 1,
    progress: ProgressLogger | None = None,
) -> BlenderMockupRecord:
    """Render the approved print on a deterministic UV-mapped 3D product mesh."""
    blender = find_blender_executable()
    if blender is None:
        return BlenderMockupRecord(
            print_path, None, None, None, "Blender", pose.name, "failed",
            "Blender was not found. Set BLENDER_EXECUTABLE or install the bundled portable renderer.", {},
        )
    stem = print_path.stem.replace("_rgb", "")
    suffix = f"_v{max(1, variant):02d}"
    preview_path = output_dir / "blender_textures" / f"{stem}{suffix}_preview.png"
    mockup_path = output_dir / "lifestyle_mockups" / f"{stem}{suffix}_lifestyle.png"
    log_path = output_dir / "blender_logs" / f"{stem}{suffix}.log"
    try:
        write_texture_preview(print_path, preview_path)
        mockup_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return BlenderMockupRecord(print_path, None, None, None, "Blender", pose.name, "failed", f"could not prepare print texture: {exc}", {})

    scene_script = Path(__file__).with_name("blender_scene.py")
    command = [
        str(blender), "--background", "--python", str(scene_script), "--",
        "--print-preview", str(preview_path),
        "--output", str(mockup_path),
        "--target", target.name,
        "--pose", pose.name,
    ]
    if progress:
        progress(f"Blender 3D render: {print_path.name} ({pose.name}).")
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND:\n" + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
    except Exception as exc:
        return BlenderMockupRecord(print_path, None, None, None, "Blender", pose.name, "failed", f"Blender render failed: {exc}", {})
    if completed.returncode != 0 or not mockup_path.exists():
        detail = tail_text(completed.stderr) or tail_text(completed.stdout) or f"Blender exited with {completed.returncode}."
        return BlenderMockupRecord(print_path, None, None, None, "Blender", pose.name, "failed", detail, {"log_path": log_path})
    return BlenderMockupRecord(
        print_path, None, None, mockup_path, "Blender", pose.name, "ok",
        "Blender rendered the approved print through a UV-mapped textile mesh.",
        {"texture_preview_path": preview_path, "log_path": log_path, "renderer": "BLENDER_EEVEE"},
        variant=variant,
    )


def find_blender_executable() -> Path | None:
    configured = os.environ.get("BLENDER_EXECUTABLE", "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[2]
    bundled = repo_root / "third_party" / "blender-5.2.1" / "blender-5.2.1-windows-x64" / "blender.exe"
    return bundled if bundled.is_file() else None


def write_texture_preview(source: Path, destination: Path, max_side: int = 2048) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(destination, "PNG", optimize=True)


def tail_text(value: str, max_lines: int = 10) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " ".join(lines[-max_lines:])
