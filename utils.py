"""
Utility functions for text normalization and WER calculation.
"""

import re
import unicodedata
from jiwer import wer, cer


def normalize_text(text: str, language: str = "en") -> str:
    """
    Normalize text for WER comparison.

    Args:
        text: Input text to normalize
        language: Language code ('en' or 'pt')

    Returns:
        Normalized text
    """
    if text is None:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Unicode normalization (handle accented characters consistently)
    text = unicodedata.normalize("NFKC", text)

    # Remove punctuation but keep apostrophes for contractions
    text = re.sub(r"[^\w\s']", " ", text)

    # Handle common transcription differences
    # Numbers to words could be added here if needed

    # Language-specific normalizations
    if language == "en":
        # Common English contractions and variations
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
        # Common Portuguese variations
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
            # Use word boundaries to avoid partial replacements
            text = re.sub(rf"\b{old}\b", new, text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Remove standalone apostrophes
    text = re.sub(r"\s'\s", " ", text)
    text = re.sub(r"^'\s", "", text)
    text = re.sub(r"\s'$", "", text)

    return text


def calculate_wer(reference: str, hypothesis: str, language: str = "en") -> float:
    """
    Calculate Word Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth transcription
        hypothesis: Model's transcription
        language: Language code for normalization

    Returns:
        WER as a float (0.0 to 1.0+)
    """
    ref_normalized = normalize_text(reference, language)
    hyp_normalized = normalize_text(hypothesis, language)

    # Handle empty strings
    if not ref_normalized:
        return 0.0 if not hyp_normalized else 1.0
    if not hyp_normalized:
        return 1.0

    return wer(ref_normalized, hyp_normalized)


def calculate_cer(reference: str, hypothesis: str, language: str = "en") -> float:
    """
    Calculate Character Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth transcription
        hypothesis: Model's transcription
        language: Language code for normalization

    Returns:
        CER as a float (0.0 to 1.0+)
    """
    ref_normalized = normalize_text(reference, language)
    hyp_normalized = normalize_text(hypothesis, language)

    # Handle empty strings
    if not ref_normalized:
        return 0.0 if not hyp_normalized else 1.0
    if not hyp_normalized:
        return 1.0

    return cer(ref_normalized, hyp_normalized)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def get_language_code(dataset_name: str) -> str:
    """
    Determine language code from dataset name.

    Args:
        dataset_name: Dataset identifier

    Returns:
        Language code ('en' or 'pt')
    """
    pt_indicators = ["pt", "portuguese", "pt_br", "brazilian"]
    dataset_lower = dataset_name.lower()

    for indicator in pt_indicators:
        if indicator in dataset_lower:
            return "pt"

    return "en"
