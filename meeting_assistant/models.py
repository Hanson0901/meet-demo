from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AudioSegment:
    path: str
    start_time: float
    end_time: float
    source: str

@dataclass(frozen=True)
class Utterance:
    meeting_id: str
    utterance_id: str
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float

