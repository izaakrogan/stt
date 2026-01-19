import re
import unicodedata
from jiwer import wer


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_wer(reference: str, hypothesis: str, language: str = "en") -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0

    return wer(ref, hyp)
