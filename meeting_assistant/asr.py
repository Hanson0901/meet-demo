from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from threading import RLock

from .logging import EventLogger
from .models import AudioSegment, Utterance


@dataclass
class _Recognition:
    start: float
    end: float
    text: str
    speaker: str
    confidence: float


@dataclass
class _FileRecognition:
    items: list[_Recognition]
    speaker_embeddings: dict[str, list[float]]


class FunASRAdapter:
    """Lazy FunASR adapter with one full-file recognition and voiceprint cache."""

    def __init__(self, logger: EventLogger):
        self.logger = logger
        self.model = None
        self.speaker_model_name = os.getenv("FUNASR_SPK_MODEL", "cam++") or "cam++"
        self._cache: dict[str, _FileRecognition] = {}
        self._model_lock = RLock()
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
            if key == "spk_model":
                self.speaker_model_name = value or "cam++"
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
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _sentence_bounds(cls, sentence: dict) -> tuple[float, float]:
        timestamps = sentence.get("timestamp") or sentence.get("timestamps") or []
        start = cls._number(sentence.get("start"))
        end = cls._number(sentence.get("end"), start)
        if timestamps and isinstance(timestamps[0], (list, tuple)):
            start = cls._number(timestamps[0][0], start)
            end = cls._number(timestamps[-1][-1], end)
        if start > 1000 or end > 1000:
            start, end = start / 1000, end / 1000
        return max(0.0, start), max(0.0, end)

    def _extract_speaker_embeddings(
        self, path: str, items: list[_Recognition]
    ) -> dict[str, list[float]]:
        """Reuse the diarization CAM++ model to get one vector per speaker."""
        speaker_model = getattr(self.model, "spk_model", None)
        if speaker_model is None or not items:
            return {}
        try:
            import numpy as np
            import torch
            from funasr.utils.load_utils import load_audio_text_image_video

            audio = load_audio_text_image_video(path, fs=16000, audio_fs=16000)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            waveform = np.asarray(audio, dtype=np.float32).squeeze()
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[1] else 1)
            if waveform.ndim != 1 or waveform.size == 0:
                return {}

            grouped: dict[str, list[np.ndarray]] = {}
            sample_budget = 12 * 16000
            lengths: dict[str, int] = {}
            for item in items:
                used = lengths.get(item.speaker, 0)
                remaining = sample_budget - used
                if remaining <= 0 or item.end <= item.start:
                    continue
                start = min(waveform.size, max(0, int(item.start * 16000)))
                end = min(waveform.size, max(start, int(item.end * 16000)))
                if end - start < int(0.35 * 16000):
                    continue
                clip = waveform[start:min(end, start + remaining)]
                if clip.size:
                    grouped.setdefault(item.speaker, []).append(clip)
                    lengths[item.speaker] = used + clip.size

            samples: list[np.ndarray] = []
            speaker_ids: list[str] = []
            for speaker_id, clips in grouped.items():
                joined = np.concatenate(clips)
                # CAM++ is less reliable on very short turns. Combine this
                # speaker's turns, but never pad a tiny sample into a match.
                if joined.size < int(1.5 * 16000):
                    continue
                samples.append(joined)
                speaker_ids.append(speaker_id)
            if not samples:
                return {}

            device = getattr(self.model, "kwargs", {}).get("device", "cpu")
            speaker_model.eval()
            with torch.no_grad():
                results, _ = speaker_model.inference(
                    data_in=samples, fs=16000, device=device
                )

            embeddings: dict[str, list[float]] = {}
            for speaker_id, result in zip(speaker_ids, results):
                vector = result.get("spk_embedding")
                if vector is None:
                    continue
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu().numpy()
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm > 1e-12 and np.isfinite(vector).all():
                    embeddings[speaker_id] = (vector / norm).astype(float).tolist()
            return embeddings
        except Exception as exc:
            # Voiceprint enrollment must not make an otherwise good transcript fail.
            self.logger.emit("ASR", f"CAM++ 聲紋特徵擷取略過: {exc}", level="WARNING")
            return {}

    def _recognize_file(self, path: str) -> _FileRecognition:
        result = self._load_model().generate(input=path, batch_size_s=300)
        if not result:
            return _FileRecognition([], {})
        first = result[0] if isinstance(result, list) else result
        sentences = first.get("sentence_info") or first.get("sentences") or []
        if not sentences:
            text = str(first.get("text", "")).strip()
            items = [_Recognition(0.0, 0.0, text, "speaker_0", 0.0)] if text else []
            return _FileRecognition(items, {})

        output = []
        for sentence in sentences:
            start, end = self._sentence_bounds(sentence)
            text = str(sentence.get("text", "")).strip()
            speaker = sentence.get("spk")
            if speaker is None:
                speaker = sentence.get("speaker", "speaker_0")
            if text:
                output.append(_Recognition(
                    start, end, text, str(speaker),
                    self._number(sentence.get("confidence", sentence.get("score"))),
                ))
        embeddings = self._extract_speaker_embeddings(path, output)
        return _FileRecognition(output, embeddings)

    def speaker_embeddings(self, path: str) -> dict[str, list[float]]:
        """Return cached normalized CAM++ embeddings for a recognized file."""
        with self._model_lock:
            result = self._cache.get(str(path))
            return dict(result.speaker_embeddings) if result else {}

    def transcribe(self, meeting_id: str, segment: AudioSegment) -> list[Utterance]:
        try:
            with self._model_lock:
                if segment.path not in self._cache:
                    self._cache[segment.path] = self._recognize_file(segment.path)
                recognized = self._cache[segment.path].items
            selected = [item for item in recognized if item.end == 0 or (item.end >= segment.start_time and item.start <= segment.end_time)]
            return [Utterance(meeting_id, str(uuid.uuid4()), item.speaker, item.start, item.end or segment.end_time, item.text, item.confidence) for item in selected]
        except RuntimeError:
            return [Utterance(meeting_id, str(uuid.uuid4()), "speaker_0", segment.start_time, segment.end_time, "[待以 FunASR 辨識音訊]", 0.0)]
        except Exception as exc:
            self.logger.emit("ASR", f"FunASR 辨識失敗: {exc}", meeting_id=meeting_id, level="ERROR")
            return [Utterance(meeting_id, str(uuid.uuid4()), "speaker_0", segment.start_time, segment.end_time, "[ASR 辨識失敗]", 0.0)]
