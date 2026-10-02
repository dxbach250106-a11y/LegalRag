from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    root: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(root=config_path.parent, data=data)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data.get(name, {}))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        value: Any = self.data
        for key in dotted_key.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def path(self, key: str) -> Path:
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"Missing configured path: paths.{key}")
        path = Path(raw)
        return path if path.is_absolute() else self.root / path
