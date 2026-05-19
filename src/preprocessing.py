import re

def preprocess_text(text: str) -> str:
    """Deterministic preprocessing used during training and inference."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
