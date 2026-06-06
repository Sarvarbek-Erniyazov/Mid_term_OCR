

import re


def postprocess_text(ocr_result: dict) -> dict:
    """
    OCR natijasini tozalaydi va statistika qaytaradi.

    Returns:
        {
          "clean_text" : str,
          "stats"      : {"words": int, "chars": int, "confidence": str, "model": str}
        }
    """
    raw = ocr_result.get("text", "")
    conf = ocr_result.get("confidence")
    model = ocr_result.get("model", "")

    clean = _normalize(raw)

    stats = {
        "words"      : len(clean.split()),
        "chars"      : len(clean),
        "confidence" : f"{conf:.2f}" if conf is not None else "N/A (TrOCR)",
        "model"      : model,
    }

    return {"clean_text": clean, "stats": stats}


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    CER — Character Error Rate .
    CER = edit_distance(ref, hyp) / len(ref)
    """
    if not reference:
        return 0.0
    return _edit_distance(reference, hypothesis) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    WER — Word Error Rate .
    WER = edit_distance(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


# ------------------------------------------------------------------
# Ichki yordamchi funksiyalar
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Bo'shliq va tinish belgilarini normallashtirish."""
    text = re.sub(r'\s+', ' ', text)            # ko'p bo'shliq → bitta
    text = re.sub(r' ([.,;:!?])', r'\1', text)  # ". " o'rniga "."
    text = text.strip()
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
