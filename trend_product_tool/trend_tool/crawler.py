from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateImage:
    path: Path
    source: str
    keyword: str
    source_role: str = "unknown"
    metadata: dict[str, object] | None = None
