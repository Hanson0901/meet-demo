from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from .logging import EventLogger
from .models import AudioSegment, Utterance

@dataclass
class _Recognition:
    start: float
    end: float
    text: str
    speaker: str
    confidence: float

class FunASRAdapter:
    """Lazy FunASR adapter with one full-file recognition cache."""
    def __init__(self, logger: EventLogger):
        self.logger = logger
        self.model = None
        self._cache: dict[str, list[_Recognition]] = {}
        self._unavailable_logged = False

    def _load_model(self):
        if self.model is not None:
            return self.model
        try:
            from funasr import AutoModel
        except ImportError as exc:
            if not self._unavailable_logged:
                self.logger.emit("ASR", "目前 Python 環境沒有 FunASR，請使用 py3_10 環境", level="ERROR")
                self._unavailable_logged = True
            raise RuntimeError("FunASR is not installed") from exc
        model_name = os.getenv("FUNASR_MODEL", "paraformer-zh")
        kwargs = {"model": model_name}
        for key, option in (("vad_model", "FUNASR_VAD_MODEL"), ("punc_model", "FUNASR_PUNC_MODEL"), ("spk_model", "FUNASR_SPK_MODEL")):
            value = os.getenv(option, {"vad_model": "fsmn-vad", "punc_model": "ct-punc-c", "spk_model": "cam++"}[key])
            if value:
                kwargs[key] = value
        if "vad_model" in kwargs:
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        self.logger.emit("ASR", f"loading FunASR model={model_name}")
        self.model = AutoModel(**kwargs)
        self.logger.emit("ASR", "FunASR model loaded")
        return self.model

    @staticmethod
    def _number(value, default=0.0):
        try: return float(value)
        except (TypeError, ValueError): return default

    def _recognize_file(self, path: str) -> list[_Recognition]:
        result = self._load_model().generate(input=path, batch_size_s=300)
        if not result: return []
        first = result[0] if isinstance(result, list) else result
        sentences = first.get("sentence_info") or first.get("sentences") or []
        if not sentences:
            text = str(first.get("text", "")).strip()
            return [_Recognition(0.0, 0.0, text, "speaker_0", 0.0)] if text else []
        output = []
        for sentence in sentences:
            timestamps = sentence.get("timestamp") or sentence.get("timestamps") or []
            start = self._number(sentence.get("start")); end = self._number(sentence.get("end"), start)
            if timestamps and isinstance(timestamps[0], (list, tuple)):
                start = self._number(timestamps[0][0], start); end = self._number(timestamps[-1][-1], end)
            if start > 1000 or end > 1000: start, end = start / 1000, end / 1000
            text = str(sentence.get("text", "")).strip()
            if text: output.append(_Recognition(start, end, text, str(sentence.get("spk", sentence.get("speaker", "speaker_0"))), self._number(sentence.get("confidence", sentence.get("score")))))
        return output

    def transcribe(self, meeting_id: str, segment: AudioSegment) -> list[Utterance]:
        try:
            if segment.path not in self._cache: self._cache[segment.path] = self._recognize_file(segment.path)
            selected = [item for item in self._cache[segment.path] if item.end == 0 or (item.end >= segment.start_time and item.start <= segment.end_time)]
            return [Utterance(meeting_id, str(uuid.uuid4()), item.speaker, item.start, item.end or segment.end_time, item.text, item.confidence) for item in selected]
        except RuntimeError:
            return [Utterance(meeting_id, str(uuid.uuid4()), "speaker_0", segment.start_time, segment.end_time, "[待以 FunASR 辨識音訊]", 0.0)]
        except Exception as exc:
            self.logger.emit("ASR", f"FunASR 辨識失敗: {exc}", meeting_id=meeting_id, level="ERROR")
            return [Utterance(meeting_id, str(uuid.uuid4()), "speaker_0", segment.start_time, segment.end_time, "[ASR 辨識失敗]", 0.0)]
