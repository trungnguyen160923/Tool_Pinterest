from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task3ReplacementConfig:
    reference_dir: Path
    output_dir: Path
    modes: tuple[str, ...] = ("standard",)
    models: tuple[str, ...] = ("nb2",)
    targets: tuple[str, ...] = ("1K",)
    output_limit: int = 0


@dataclass(frozen=True)
class Task3ReplacementResult:
    artwork_path: Path
    run_dir: Path | None
    outputs: list[Path]
    log_path: Path
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_task3_replacements(artworks: list[Path], config: Task3ReplacementConfig | None) -> list[Task3ReplacementResult]:
    if config is None or not config.reference_dir.exists():
        return []
    selected = artworks[: config.output_limit] if config.output_limit > 0 else artworks[:1]
    return [run_one_task3_replacement(artwork, config) for artwork in selected]


def run_one_task3_replacement(artwork: Path, config: Task3ReplacementConfig) -> Task3ReplacementResult:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "task3_image_replace" / "benchmark_rug_artwork_replacement.py"
    if not script.exists():
        raise RuntimeError(f"task3 replacement script not found: {script}")

    stage_root = config.output_dir / artwork.stem
    stage_root.mkdir(parents=True, exist_ok=True)
    before = existing_dirs(stage_root)
    command = [
        sys.executable,
        str(script),
        "--artwork-file",
        str(artwork),
        "--reference-dir",
        str(config.reference_dir),
        "--output-root",
        str(stage_root),
        "--modes",
        *config.modes,
        "--models",
        *config.models,
        "--targets",
        *config.targets,
        "--max-refs",
        "1",
        "--format",
        "png",
    ]
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, input="y\n")
    log_path = stage_root / "task3_replacement.log"
    write_process_log(log_path, command, completed)
    run_dir = newest_new_dir(stage_root, before)
    outputs = sorted((run_dir / "outputs").glob("*.*")) if run_dir and (run_dir / "outputs").exists() else []
    status = "ok" if completed.returncode == 0 and outputs else "failed"
    notes = "AI artwork replacement completed." if status == "ok" else tail_text(completed.stderr) or tail_text(completed.stdout)
    return Task3ReplacementResult(artwork, run_dir, outputs, log_path, status, notes)


def existing_dirs(path: Path) -> set[Path]:
    return {item for item in path.iterdir() if item.is_dir()} if path.exists() else set()


def newest_new_dir(path: Path, before: set[Path]) -> Path | None:
    after = [item for item in path.iterdir() if item.is_dir() and item not in before]
    if not after:
        return None
    return max(after, key=lambda item: item.stat().st_mtime)


def write_process_log(path: Path, command: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "COMMAND:",
                " ".join(command),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )


def tail_text(value: str, max_lines: int = 8) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " ".join(lines[-max_lines:])
