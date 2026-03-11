from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

class RunMessages:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def has_messages(self) -> bool:
        return bool(self.warnings or self.infos)
