"""
app.py
======
Gradio UI — faqat interfeys.
Barcha mantiq pipeline/ modullarida.

Yangiliklar:
  - PDF qabul qilish
  - CLAHE preprocessing
  - Spell check
  - CER/WER ko'rsatish
"""

import gradio as gr
import tempfile
import os
from PIL import Image

from pipeline import preprocess_image, run_ocr, postprocess_text, analyze_report
from pipeline.ocr import run_ocr_pdf


# ---------------------------------------------------------------
# Rasm pipeline
# ---------------------------------------------------------------
def process_image(image_path, model_choice, dars_nomi):
    if image_path is None:
        return "Rasm yuklanmadi.", "", "", ""

    processed_image = preprocess_image(image_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        processed_image.save(tmp.name)

    ocr_result = run_ocr(processed_image, model_choice)
    post_result = postprocess_text(ocr_result)
    clean_text = post_result["clean_text"]
    stats = post_result["stats"]

    stats_str = (
        f"Model: {stats['model']}  |  "
        f"So'z: {stats['words']}  |  "
        f"Belgi: {stats['chars']}  |  "
        f"Confidence: {stats['confidence']}"
    )

    report = analyze_report(clean_text, dars_nomi)
    return tmp.name, clean_text, stats_str, report


# ---------------------------------------------------------------
# PDF pipeline
# ---------------------------------------------------------------
def process_pdf(pdf_file, model_choice, dars_nomi):
    if pdf_file is None:
        return "PDF yuklanmadi.", "", ""

    result = run_ocr_pdf(pdf_file.name, model_choice)
    clean_text = result["text"]
    pages = result.get("pages", "?")
    conf = result.get("confidence")

    stats_str = (
        f"Model: {result['model']}  |  "
        f"Sahifalar: {pages}  |  "
        f"Confidence: {conf if conf else 'N/A'}"
    )

    report = analyze_report(clean_text, dars_nomi)
    return clean_text, stats_str, report


# ---------------------------------------------------------------
# Gradio interfeysi
# ---------------------------------------------------------------
with gr.Blocks(title="Qo'l Yozma Report Baholash") as demo:

    gr.Markdown("""
    # 📝 Qo'l Yozma Report Baholash Tizimi
    **Computer Vision Kursi — Dars 5 | Oraliq Nazorat**

    Rasm yoki PDF yuklang → tizim o'qiydi → **ustozga baho + xulosa** beradi.
    """)

    with gr.Tabs():

        # ---- Tab 1: Rasm ----
        with gr.TabItem("📷 Rasm"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="filepath", label="Report rasmi")
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
                    run_btn = gr.Button("🔍 O'qish va baholash", variant="primary")

                with gr.Column():
                    preprocessed_out = gr.Image(label="1️⃣ Preprocessing natijasi")
                    ocr_out = gr.Textbox(label="2️⃣ OCR + Spell Check natijasi", lines=5)
                    stats_out = gr.Textbox(label="📊 Statistika", interactive=False)
                    report_out = gr.Textbox(label="3️⃣ Ustozga hisobot + baho", lines=10)

            run_btn.click(
                fn=process_image,
                inputs=[img_input, model_radio, dars_input],
                outputs=[preprocessed_out, ocr_out, stats_out, report_out],
            )

        # ---- Tab 2: PDF ----
        with gr.TabItem("📄 PDF"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="PDF faylni yuklang", file_types=[".pdf"])
                    pdf_model = gr.Radio(
                        choices=["EasyOCR", "TrOCR"],
                        value="EasyOCR",
                        label="OCR modeli"
                    )
                    pdf_dars = gr.Textbox(
                        label="Dars nomi (ixtiyoriy)",
                        placeholder="masalan: Dars 5 — Advanced OCR",
                    )
                    pdf_btn = gr.Button("🔍 PDF O'qish va baholash", variant="primary")

                with gr.Column():
                    pdf_ocr_out = gr.Textbox(label="2️⃣ OCR natijasi (barcha sahifalar)", lines=8)
                    pdf_stats_out = gr.Textbox(label="📊 Statistika", interactive=False)
                    pdf_report_out = gr.Textbox(label="3️⃣ Ustozga hisobot + baho", lines=10)

            pdf_btn.click(
                fn=process_pdf,
                inputs=[pdf_input, pdf_model, pdf_dars],
                outputs=[pdf_ocr_out, pdf_stats_out, pdf_report_out],
            )

    gr.Markdown("""
    ---
    💡 **Maslahat:** rasm yorug', to'g'ri burchakdan olingan bo'lsa OCR aniqroq ishlaydi.
    """)


if __name__ == "__main__":
    demo.launch()
