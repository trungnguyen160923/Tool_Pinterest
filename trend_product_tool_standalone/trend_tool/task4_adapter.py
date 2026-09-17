from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class Task4MockupConfig:
    output_dir: Path
    background: str = ""
    modes: tuple[str, ...] = ("flex",)
    models: tuple[str, ...] = ("pro",)
    final_integration: str = "on"
    limit: int = 1


@dataclass(frozen=True)
class Task4MockupResult:
    product_path: Path
    run_dir: Path | None
    outputs: list[Path]
    log_path: Path
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_task4_mockups(
    products: list[Path],
    config: Task4MockupConfig | None,
    progress: ProgressLogger | None = None,
    mask_files: dict[Path, Path] | None = None,
) -> list[Task4MockupResult]:
    if config is None:
        return []
    selected = products[: max(1, config.limit)]
    masks = mask_files or {}
    return [run_one_task4_mockup(product, config, progress, masks.get(product)) for product in selected]


def run_one_task4_mockup(
    product: Path,
    config: Task4MockupConfig,
    progress: ProgressLogger | None = None,
    mask_file: Path | None = None,
) -> Task4MockupResult:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "task4_background_replace" / "benchmark_product_background_replace_semantic_v4_1_best_practice.py"
    if not script.exists():
        raise RuntimeError(f"task4 background replacement script not found: {script}")

    stage_root = config.output_dir / product.stem
    stage_root.mkdir(parents=True, exist_ok=True)
    before = existing_dirs(stage_root)
    command = [
        sys.executable,
        str(script),
        "--product-file",
        str(product),
        "--output-root",
        str(stage_root),
        "--modes",
        *config.modes,
        "--models",
        *config.models,
        "--final-integration",
        config.final_integration,
        "--realism-strength",
        "1.0",
        "--edge-light-wrap",
        "0.95",
        "--shadow-strength",
        "0.38",
        "--yes",
    ]
    if config.background.strip():
        command.extend(["--background", config.background.strip()])
    if mask_file is not None:
        command.extend(["--mask-file", str(mask_file)])

    log_path = stage_root / "task4_mockup.log"
    completed = run_process(command, repo_root, log_path, progress)
    run_dir = newest_new_dir(stage_root, before)
    outputs = sorted(run_dir.rglob("*final*.png")) if run_dir else []
    if not outputs and run_dir:
        outputs = sorted(run_dir.rglob("*.png"))
    status = "ok" if completed.returncode == 0 and outputs else "failed"
    notes = "AI background replacement completed." if status == "ok" else tail_text(completed.stderr) or tail_text(completed.stdout)
    return Task4MockupResult(product, run_dir, outputs, log_path, status, notes)


def run_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    progress: ProgressLogger | None,
) -> subprocess.CompletedProcess[str]:
    if progress is None:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        write_process_log(log_path, command, completed)
        return completed

    stdout_lines: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command))
        log_file.write("\n\nSTDOUT/STDERR:\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            stdout_lines.append(raw_line)
            log_file.write(raw_line)
            log_file.flush()
            line = raw_line.strip()
            if line:
                progress(f"AI background: {line}")
        returncode = process.wait()
    stdout = "".join(stdout_lines)
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


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
