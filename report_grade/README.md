# 📝 Qo'l Yozma Report Baholash Tizimi

**Computer Vision Kursi — Dars 5 | Oraliq Nazorat (Mid-term)**

---

## Loyiha tuzilmasi

```
report_grader/
│
├── app.py                   # Gradio UI — faqat interfeys
│
├── pipeline/
│   ├── __init__.py          # modulni import qilish
│   ├── preprocess.py        # Dars 2: Grayscale→Denoise→Binarize→Deskew→Morph
│   ├── ocr.py               # Dars 1,3,5: EasyOCR + TrOCR (line split bilan)
│   ├── postprocess.py       # Dars 5: normalize, CER/WER metrics
│   └── llm.py               # Gemini tahlil + baholash
│
├── requirements.txt
├── .env.example             # GEMINI_API_KEY namunasi
└── README.md
```

---

## Pipeline (qadamlar va darslar)

```
Rasm
 │
 ├─ preprocess.py  (Dars 2)
 │    Grayscale → GaussianBlur → Otsu → Deskew → Morphology
 │
 ├─ ocr.py         (Dars 1, 3, 5)
 │    EasyOCR  — tez, lotin/kirill
 │    TrOCR   — handwriting, satrga bo'lib o'qiydi
 │
 ├─ postprocess.py (Dars 5)
 │    Normalize, confidence statistika, CER/WER
 │
 └─ llm.py         (yangi)
      Gemini → o'zbekcha xulosa + baho
```

---

## O'rnatish

### 1. Kutubxonalar
```bash
pip install -r requirements.txt
```

### 2. Gemini API kalit (bepul)
1. https://aistudio.google.com/app/apikey — kalit oling
2. `.env.example` → `.env` deb nusxa oling
3. Kalitingizni yozing: `GEMINI_API_KEY=...`

### 3. Ishga tushirish
```bash
python app.py
```

---

## Hugging Face Spaces'ga deploy

1. https://huggingface.co/new-space — yangi Space oching
2. **SDK: Gradio** tanlang
3. `app.py` va `requirements.txt` ni yuklang
4. Settings → **Secrets** → `GEMINI_API_KEY` qo'shing
5. Tayyor — ustoz istalgan vaqt brauzerdan ishlatadi

---

## Savol-javobga tayyorlanish

| Savol | Javob |
|-------|-------|
| TrOCR vs EasyOCR? | TrOCR — Transformer (ViT+decoder), handwriting'da eng aniq lekin sekin. EasyOCR — deep learning, tez, ko'p tilli. |
| Preprocessing nima beradi? | OCR aniqligini 30-50% oshiradi (Dars 3). |
| CER/WER nima? | Character/Word Error Rate — OCR sifatini o'lchaydi (Dars 5). |
| Confidence filter nima? | Past ishonchli (< 0.30) bloklarni tashlab, xato o'qishlarni kamaytiradi. |
| Nima uchun LLM kerak? | OCR matnni o'qiydi; LLM tushunadi, mavzularni aniqlaydi, baholaydi. |
| Modular qilish foydasi? | Har modul alohida testlanadi, almashtirish oson (masalan TrOCR→PaddleOCR). |
