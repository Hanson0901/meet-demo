"""Dependency-free hashed n-gram vectors for local meeting retrieval."""
from __future__ import annotations
import hashlib
import math
import re

DIMENSIONS = 512

def _features(text: str):
    text = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text)
    features = []
    for word in words:
        if word.isascii() and len(word) > 1:
            features.append(word)
        elif not word.isascii():
            features.append(word)
    cjk = "".join(word for word in words if not word.isascii())
    for size in (2, 3, 4):
        features.extend(cjk[index:index + size] for index in range(max(0, len(cjk) - size + 1)))
    return features

def embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % DIMENSIONS
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector

def similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

def transcript_chunks(rows: list[dict], *, max_chars: int = 1400) -> list[str]:
    chunks, current = [], []
    length = 0
    for row in rows:
        line = f"{row.get('speaker_name', row['speaker_id'])}: {row.get('normalized_text', row.get('text', ''))}"
        if current and length + len(line) > max_chars:
            chunks.append("\n".join(current)); current, length = [], 0
        current.append(line); length += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks
