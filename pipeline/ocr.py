
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
    """
    EasyOCR bilan o'qish.
    Confidence filter: 0.30 dan past bloklarni tashlaydi.
    """
    reader = _get_easy_reader()

    # PIL → temp fayl (EasyOCR fayl yo'li kutadi)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        results = reader.readtext(tmp.name)   # [(bbox, text, conf), ...]

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
        name = "microsoft/trocr-base-handwritten"
        _trocr_processor = TrOCRProcessor.from_pretrained(name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(name)
    return _trocr_processor, _trocr_model


def run_ocr_trocr(image: Image.Image) -> dict:
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


def _split_into_lines(image: Image.Image, min_height: int = 15) -> list:
    """
    Rasmni matn satrlariga bo'ladi (horizontal projection profile).
    Natija: [PIL Image, ...] — har biri bitta satr.
    """
    import cv2

    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_sums = binary.sum(axis=1)           
    in_text = row_sums > binary.shape[1] * 0.01   

    # Uzluksiz matn bloklarini topish
    regions, start = [], None
    for i, val in enumerate(in_text):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_height:
                regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(in_text)))

    if not regions:
        return [image]   

    h_padding = 4
    orig = np.array(image)
    line_images = []
    for (y1, y2) in regions:
        y1c = max(0, y1 - h_padding)
        y2c = min(orig.shape[0], y2 + h_padding)
        line_images.append(Image.fromarray(orig[y1c:y2c]))

    return line_images



def run_ocr(image: Image.Image, model_choice: str = "EasyOCR") -> dict:
    """
    model_choice: "EasyOCR" | "TrOCR"
    Returns: {"text": str, "confidence": float|None, "model": str}
    """
    if model_choice == "TrOCR":
        return run_ocr_trocr(image)
    return run_ocr_easyocr(image)
