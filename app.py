"""
app.py
======
Gradio UI — faqat interfeys.
Barcha mantiq pipeline/ modullarida.

Ishga tushirish:
    python app.py
    # yoki Hugging Face Spaces uchun ham shu fayl ishlatiladi
"""

import gradio as gr
import tempfile
from PIL import Image

from pipeline import preprocess_image, run_ocr, postprocess_text, analyze_report


# ---------------------------------------------------------------
# Asosiy pipeline funksiyasi (Gradio chaqiradi)
# ---------------------------------------------------------------
def process(image_path, model_choice, dars_nomi):
    if image_path is None:
        return "Rasm yuklanmadi.", "", "", ""

    # 1. Preprocessing (Dars 2)
    processed_image = preprocess_image(image_path)

    # Preprocessdan o'tgan rasmni vaqtinchalik saqlash (OCR uchun)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        processed_image.save(tmp.name)
        preprocessed_path = tmp.name

    # 2. OCR (Dars 1, 3, 5)
    ocr_result = run_ocr(processed_image, model_choice)

    # 3. Post-processing (Dars 5)
    post_result = postprocess_text(ocr_result)
    clean_text = post_result["clean_text"]
    stats = post_result["stats"]

    stats_str = (
        f"Model: {stats['model']}  |  "
        f"So'z: {stats['words']}  |  "
        f"Belgi: {stats['chars']}  |  "
        f"Confidence: {stats['confidence']}"
    )

    # 4. LLM tahlil — Gemini (llm.py)
    report = analyze_report(clean_text, dars_nomi)

    return preprocessed_path, clean_text, stats_str, report


# ---------------------------------------------------------------
# Gradio interfeysi (Dars 4 uslubida)
# ---------------------------------------------------------------
with gr.Blocks(title="Qo'l Yozma Report Baholash") as demo:

    gr.Markdown("""
    # 📝 Qo'l Yozma Report Baholash Tizimi
    **Computer Vision Kursi — Dars 5 | Oraliq Nazorat**

    Talaba qo'lda yozgan dars reportini yuklang →
    tizim o'qiydi, tahlil qiladi → **ustozga baho + xulosa** beradi.
    """)

    with gr.Row():

        # --- Chap ustun: Input ---
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="📷 Report rasmi")
            model_radio = gr.Radio(
                choices=["EasyOCR", "TrOCR"],
                value="EasyOCR",
                label="OCR modeli",
                info="EasyOCR — tez | TrOCR — handwriting uchun aniqroq"
            )
            dars_input = gr.Textbox(
                label="Dars nomi (ixtiyoriy)",
                placeholder="masalan: Dars 3 — Preprocessing",
            )
            run_btn = gr.Button("🔍 O'qish va baholash", variant="primary", size="lg")

        # --- O'ng ustun: Output ---
        with gr.Column(scale=1):
            preprocessed_out = gr.Image(label="1️⃣ Preprocessing natijasi")
            ocr_out = gr.Textbox(label="2️⃣ OCR — o'qilgan matn", lines=5)
            stats_out = gr.Textbox(label="📊 Statistika", interactive=False)
            report_out = gr.Textbox(label="3️⃣ Ustozga hisobot + baho", lines=10)

    run_btn.click(
        fn=process,
        inputs=[img_input, model_radio, dars_input],
        outputs=[preprocessed_out, ocr_out, stats_out, report_out],
    )

    gr.Markdown("""
    ---
    💡 **Maslahat:** rasm yorug', to'g'ri burchakdan olingan bo'lsa OCR aniqroq ishlaydi.
    """)


if __name__ == "__main__":
    demo.launch()
