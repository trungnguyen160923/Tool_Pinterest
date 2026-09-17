from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrendCandidate:
    candidate_id: str
    name: str
    source: str
    rank: int
    strength: float
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

