from __future__ import annotations
import wave
from pathlib import Path
from .logging import EventLogger
from .models import AudioSegment

class AudioIngest:
    def __init__(self, logger: EventLogger): self.logger = logger
    def segments(self, path: str, *, chunk_seconds: int = 30):
        source = str(Path(path)); duration = 0.0
        try:
            with wave.open(source) as audio: duration = audio.getnframes() / max(audio.getframerate(), 1)
        except (wave.Error, OSError):
            self.logger.emit("AUDIO", "非 WAV 音訊，保留原檔交由 ASR adapter 處理", level="WARNING")
        if duration <= 0: duration = float(chunk_seconds)
        start = 0.0
        while start < duration:
            end = min(start + chunk_seconds, duration)
            yield AudioSegment(source, start, end, "file")
            start = end

