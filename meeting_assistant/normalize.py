_MAP = str.maketrans({"臺":"台", "裡":"裡", "裏":"裡", "發":"發", "後":"後"})
def convert_traditional(text: str) -> str:
    """Stable local hook; full zh-cn -> zh-tw conversion can be injected later."""
    return text.translate(_MAP)

