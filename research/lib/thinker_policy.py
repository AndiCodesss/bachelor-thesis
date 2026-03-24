from __future__ import annotations

from typing import Any


class ThinkerPolicyError(ValueError):
    def __init__(self, message: str, *, brief: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.brief = dict(brief or {})


__all__ = ["ThinkerPolicyError"]
