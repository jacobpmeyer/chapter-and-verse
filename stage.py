"""The result of a paid indexing stage (summaries or embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Progress = Callable[[str], None]


@dataclass
class StageResult:
    ok: bool
    cost_usd: float

    def __bool__(self) -> bool:  # `if not summarize.execute(...)` keeps working
        return self.ok


def report(progress: Progress | None, message: str) -> None:
    """Print a progress line, and pass it to the caller (e.g. a job's status) if it asked."""
    print(message, flush=True)
    if progress:
        progress(message.strip())
