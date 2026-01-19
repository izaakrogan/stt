import re
import unicodedata
from jiwer import wer, cer


def normalize_text(text: str, language: str = "en") -> str:
    if text is None:
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s']", " ", text)

    if language == "en":
        replacements = {
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of",
            "dunno": "don't know",
            "lemme": "let me",
            "gimme": "give me",
            "'cause": "because",
            "okay": "ok",
            "yeah": "yes",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

    elif language == "pt":
        replacements = {
            "pra": "para",
            "pro": "para o",
            "tá": "está",
            "tô": "estou",
            "né": "não é",
            "vc": "você",
            "tb": "também",
        }
        for old, new in replacements.items():
            text = re.sub(rf"\b{old}\b", new, text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"\s'\s", " ", text)
    text = re.sub(r"^'\s", "", text)
    text = re.sub(r"\s'$", "", text)

    return text


def calculate_wer(reference: str, hypothesis: str, language: str = "en") -> float:
    ref_normalized = normalize_text(reference, language)
    hyp_normalized = normalize_text(hypothesis, language)

    if not ref_normalized:
        return 0.0 if not hyp_normalized else 1.0
    if not hyp_normalized:
        return 1.0

    return wer(ref_normalized, hyp_normalized)


def calculate_cer(reference: str, hypothesis: str, language: str = "en") -> float:
    ref_normalized = normalize_text(reference, language)
    hyp_normalized = normalize_text(hypothesis, language)

    if not ref_normalized:
        return 0.0 if not hyp_normalized else 1.0
    if not hyp_normalized:
        return 1.0

    return cer(ref_normalized, hyp_normalized)


