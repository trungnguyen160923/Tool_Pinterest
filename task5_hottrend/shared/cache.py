from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json


class JsonCache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: dict[str, Any] = {}
        if enabled and path.exists():
            try:
                raw = read_json(path)
                if isinstance(raw, dict):
                    self.data = raw
            except Exception:
                self.data = {}

    def get(self, key: str) -> Any:
        if not self.enabled:
            return None
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        if self.enabled:
            self.data[key] = value

    def save(self) -> None:
        if self.enabled:
            write_json(self.path, self.data)

