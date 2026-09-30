"""Small synchronous event logger shared by every layer."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

@dataclass(frozen=True)
class LogEvent:
    module: str
    message: str
    level: str = "INFO"
    meeting_id: str | None = None
    timestamp: str = ""

    def line(self) -> str:
        stamp = self.timestamp or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        suffix = f" meeting={self.meeting_id}" if self.meeting_id else ""
        return f"[{self.module}] {stamp} {self.level.lower()}{suffix} {self.message}"

class EventLogger:
    def __init__(self, sink: Callable[[LogEvent], None] | None = None):
        self.sink = sink or (lambda event: print(event.line(), flush=True))

    def emit(self, module: str, message: str, *, level: str = "INFO", meeting_id: str | None = None) -> None:
        self.sink(LogEvent(module, message, level, meeting_id, datetime.now(timezone.utc).isoformat(timespec="milliseconds")))
