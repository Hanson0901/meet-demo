from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Iterable

from .normalize import convert_traditional


_FOLLOWING_NAME = (
    r"(?:\s|[,，。！？、:：]|老師|教授|醫師|醫生|博士|主任|院長|經理|協理|組長|"
    r"學長|學姐|學弟|學妹|大家好|很高興|請多指教|目前|今年|負責|來自|在|是|也|將|會|$)"
)
_SELF_INTRO = re.compile(
    r"(?:我的名字(?:是|叫)|我的姓名(?:是|叫)|我叫|名字叫|姓名是|我是)\s*"
    r"(?P<name>[\u3400-\u9fff]{2,4})(?=" + _FOLLOWING_NAME + r")"
)
_ADDRESS = re.compile(
    r"(?P<name>[\u3400-\u9fff]{2,4})"
    r"(?:老師|教授|醫師|醫生|博士|主任|院長|經理|協理|組長|學長|學姐|學弟|學妹)"
    r"[，,、\s]*(?=(?:你|您|請問|想請教|能否|可不可以|可以|覺得|認為|看法|怎麼|有沒有))"
)
_NON_NAME_WORDS = {
    "主持人", "負責人", "發言人", "專案組", "研究組", "小組長", "大家好",
    "我們", "你們", "他們", "今天", "目前", "這邊", "這裡", "本次", "本人",
    "負責", "經理人", "主管", "教授們", "同學們", "老師們", "醫師們",
}


def normalize_embedding(values: Iterable[float] | None) -> list[float] | None:
    """Return a finite, unit-length embedding suitable for local cosine matching."""
    if values is None or isinstance(values, (str, bytes)):
        return None
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not vector or len(vector) > 4096 or any(not math.isfinite(value) for value in vector):
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return None
    return [value / norm for value in vector]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = normalize_embedding(left), normalize_embedding(right)
    if a is None or b is None or len(a) != len(b):
        return -1.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def voiceprint_match_threshold() -> float:
    try:
        threshold = float(os.getenv("FUNASR_SPK_MATCH_THRESHOLD", "0.74"))
    except (TypeError, ValueError):
        threshold = 0.74
    return min(0.99, max(0.50, threshold))


def suggest_names_from_transcript(rows: list[dict]) -> dict[str, dict[str, str | float]]:
    """Suggest names only from explicit self-introductions or direct forms of address."""
    ordered = sorted(rows, key=lambda row: (float(row.get("start_time", 0) or 0), str(row.get("speaker_id", ""))))
    candidates: dict[str, dict[str, list[tuple[float, str, str]]]] = defaultdict(lambda: defaultdict(list))

    for index, row in enumerate(ordered):
        speaker_id = str(row.get("speaker_id", ""))
        text = convert_traditional(str(row.get("normalized_text") or row.get("text") or "")).strip()
        if not speaker_id or not text or text.startswith("["):
            continue
        for match in _SELF_INTRO.finditer(text):
            name = match.group("name").strip()
            if name not in _NON_NAME_WORDS:
                candidates[speaker_id][name].append((0.94, "self_introduction", text[:220]))

        for match in _ADDRESS.finditer(text):
            name = match.group("name").strip()
            if name in _NON_NAME_WORDS:
                continue
            source_row = float(row.get("start_time", 0) or 0)
            for following in ordered[index + 1 :]:
                target_id = str(following.get("speaker_id", ""))
                if not target_id or target_id == speaker_id:
                    continue
                target_start = float(following.get("start_time", source_row) or source_row)
                if target_start - source_row > 20:
                    break
                candidates[target_id][name].append((0.58, "addressed_by_name", text[:220]))
                break

    suggestions: dict[str, dict[str, str | float]] = {}
    for speaker_id, names in candidates.items():
        ranked = sorted(
            names.items(),
            key=lambda item: (max(value[0] for value in item[1]), len(item[1])),
            reverse=True,
        )
        if len(ranked) > 1:
            first_score = max(value[0] for value in ranked[0][1])
            second_score = max(value[0] for value in ranked[1][1])
            if first_score == second_score and len(ranked[0][1]) == len(ranked[1][1]):
                continue
        name, evidence_items = ranked[0]
        best = max(evidence_items, key=lambda item: item[0])
        confidence = min(0.98, best[0] + 0.02 * (len(evidence_items) - 1))
        suggestions[speaker_id] = {
            "name": name,
            "source": best[1],
            "evidence": best[2],
            "confidence": confidence,
        }
    return suggestions
