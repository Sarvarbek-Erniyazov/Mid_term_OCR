"""
pipeline/preprocess.py
======================
Dars 2 mavzusi: Preprocessing
  Grayscale → Denoise (GaussianBlur) → Binarize (Otsu) → Deskew → Morphology

Kirish : rasm fayl yo'li (str)
Chiqish: tozalangan PIL Image
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path: str) -> Image.Image:
    """
    To'liq preprocessing pipeline (Dars 2).
    Returns: PIL Image (qayta ishlangan, OCR ga tayyor)
    """
    img = cv2.imread(image_path)

    # 1. Grayscale — rangli → kulrang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise — shovqinni kamaytirish
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Binarize — Otsu thresholding (matnni fondan ajratish)
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Deskew — qiyshiq hujjatni to'g'rilash
    straightened = _deskew(binary)

    # 5. Morphological opening — mayda shovqin nuqtalarini o'chirish
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(straightened, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(cleaned)


# ------------------------------------------------------------------
# Ichki yordamchi funksiya
# ------------------------------------------------------------------
def _deskew(binary: np.ndarray) -> np.ndarray:
    """Rasm burchagini topib, to'g'rilaydi (Hough transform asosida)."""
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return binary  # matn juda kam — o'zgartirmasdan qaytaradi

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return binary  # burchak kichik — deskew kerak emas

    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated
