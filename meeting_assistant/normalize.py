"""Traditional Chinese normalization for ASR and generated meeting text."""

try:
    from opencc import OpenCC
except ImportError:  # Keep non-ASR utility commands usable before dependencies are installed.
    OpenCC = None


_CONVERTER = OpenCC("s2twp") if OpenCC is not None else None
_FALLBACK_MAP = str.maketrans({"臺": "台", "裏": "裡"})


def convert_traditional(text: str) -> str:
    """Convert Simplified Chinese to Taiwan Traditional Chinese when OpenCC is available."""
    value = str(text or "")
    if _CONVERTER is not None:
        return _CONVERTER.convert(value)
    return value.translate(_FALLBACK_MAP)
