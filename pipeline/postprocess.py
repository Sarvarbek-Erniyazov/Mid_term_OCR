"""
pipeline/postprocess.py
=======================
Dars 5 mavzusi: Post-processing
  - Matnni normallashtirish
  - Confidence filter
  - Spell check (PySpellChecker)
  - CER/WER hisoblash

Kirish : ocr.py dan kelgan dict {"text", "confidence", "model"}
Chiqish: dict {"clean_text": str, "stats": dict}
"""

import re


def postprocess_text(ocr_result: dict) -> dict:
    """
    OCR natijasini tozalaydi va statistika qaytaradi.
    """
    raw = ocr_result.get("text", "")
    conf = ocr_result.get("confidence")
    model = ocr_result.get("model", "")

    # 1. Normalize
    clean = _normalize(raw)

    # 2. Spell check
    clean = _spell_check(clean)

    stats = {
        "words"      : len(clean.split()),
        "chars"      : len(clean),
        "confidence" : f"{conf:.2f}" if conf is not None else "N/A (TrOCR)",
        "model"      : model,
    }

    return {"clean_text": clean, "stats": stats}


def compute_cer(reference: str, hypothesis: str) -> float:
    """CER — Character Error Rate (Dars 5 metric)."""
    if not reference:
        return 0.0
    return _edit_distance(reference, hypothesis) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """WER — Word Error Rate (Dars 5 metric)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _normalize(text: str) -> str:
    """Bo'shliq va tinish belgilarini normallashtirish."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    return text.strip()


def _spell_check(text: str) -> str:
    """
    PySpellChecker bilan xato so'zlarni tuzatish.
    Faqat inglizcha so'zlar uchun ishlaydi.
    O'zbek so'zlari o'zgartirilmaydi.
    """
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        words = text.split()
        corrected = []
        for word in words:
            # Faqat lotin harflardan iborat so'zlarni tekshiramiz
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) > 2:
                candidates = spell.candidates(clean_word)
                if candidates:
                    correction = spell.correction(clean_word)
                    if correction and correction != clean_word:
                        word = word.replace(clean_word, correction)
            corrected.append(word)
        return ' '.join(corrected)
    except ImportError:
        return text


def _edit_distance(a, b) -> int:
    """Levenshtein masofasi (str yoki list uchun)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]
