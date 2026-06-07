"""
pipeline/ocr.py
===============
Dars 1, 3, 5 mavzusi: Pretrained OCR modellari
  - EasyOCR  : tez, ko'p tilli, lotin/kirill
  - TrOCR    : Transformer, handwriting uchun eng aniq
  - PDF      : ko'p sahifali PDF ni o'qish

Kirish : PIL Image yoki PDF fayl yo'li
Chiqish: dict  { "text": str, "confidence": float | None, "model": str }
"""

import tempfile
import numpy as np
from PIL import Image

# ---------- EasyOCR (lazy load) ----------
_easy_reader = None

def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        _easy_reader = easyocr.Reader(["en", "ru"], gpu=False)
    return _easy_reader


def run_ocr_easyocr(image: Image.Image) -> dict:
    """EasyOCR bilan o'qish. Confidence filter: 0.30 dan past bloklarni tashlaydi."""
    reader = _get_easy_reader()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        results = reader.readtext(tmp.name)

    good = [(text, conf) for (_, text, conf) in results if conf >= 0.30]

    if not good:
        return {"text": "", "confidence": 0.0, "model": "EasyOCR"}

    combined_text = " ".join(t for t, _ in good)
    avg_conf = sum(c for _, c in good) / len(good)

    return {"text": combined_text, "confidence": round(avg_conf, 3), "model": "EasyOCR"}


# ---------- TrOCR (lazy load) ----------
_trocr_processor = None
_trocr_model = None

def _get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import os
        # Fine-tuned model bor bo'lsa uni ishlatamiz
        local_model = "trocr-uzbek"
        if os.path.exists(local_model):
            name = local_model
        else:
            name = "microsoft/trocr-base-handwritten"
        _trocr_processor = TrOCRProcessor.from_pretrained(name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(name)
    return _trocr_processor, _trocr_model


def run_ocr_trocr(image: Image.Image) -> dict:
    """TrOCR bilan o'qish. Rasmni 5 bo'lakka bo'lib o'qiydi."""
    processor, model = _get_trocr()
    rgb = image.convert("RGB")
    w, h = rgb.size
    texts = []
    step = h // 5

    for i in range(5):
        y1 = i * step
        y2 = min((i + 1) * step, h)
        crop = rgb.crop((0, y1, w, y2))
        pixels = processor(images=crop, return_tensors="pt").pixel_values
        ids = model.generate(pixels, max_new_tokens=128)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            texts.append(text)

    return {"text": "\n".join(texts), "confidence": None, "model": "TrOCR"}


# ---------- PDF o'qish ----------
def run_ocr_pdf(pdf_path: str, model_choice: str = "EasyOCR") -> dict:
    """
    PDF faylni sahifalarga bo'lib, har sahifani OCR qiladi.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return {"text": "pdf2image o'rnatilmagan: pip install pdf2image", "confidence": None, "model": model_choice}

    import os
    poppler_path = None
    # Windows uchun poppler yo'lini avtomatik topish
    possible_paths = [
        r"C:\Users\sarva\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin",
        r"C:\Program Files\poppler\bin",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            poppler_path = p
            break

    try:
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    except Exception as e:
        return {"text": f"PDF o'qib bo'lmadi: {e}", "confidence": None, "model": model_choice}

    all_texts = []
    total_conf = []

    for i, page in enumerate(pages):
        from pipeline.preprocess import preprocess_image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page.save(tmp.name)
            processed = preprocess_image(tmp.name)

        if model_choice == "TrOCR":
            result = run_ocr_trocr(processed)
        else:
            result = run_ocr_easyocr(processed)

        if result["text"].strip():
            all_texts.append(f"[Sahifa {i+1}]\n{result['text']}")
        if result["confidence"]:
            total_conf.append(result["confidence"])

    avg_conf = sum(total_conf) / len(total_conf) if total_conf else None
    return {
        "text": "\n\n".join(all_texts),
        "confidence": round(avg_conf, 3) if avg_conf else None,
        "model": model_choice,
        "pages": len(pages)
    }


# ---------- Umumiy kirish nuqtasi ----------
def run_ocr(image: Image.Image, model_choice: str = "EasyOCR") -> dict:
    """
    model_choice: "EasyOCR" | "TrOCR"
    Returns: {"text": str, "confidence": float|None, "model": str}
    """
    if model_choice == "TrOCR":
        return run_ocr_trocr(image)
    return run_ocr_easyocr(image)
