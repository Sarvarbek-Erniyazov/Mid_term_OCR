"""
pipeline/llm.py
===============
LLM tahlil: Google Gemini (bepul tier)
  - Matnni o'zbek tilida tahlil qiladi
  - Ustozga: mavzular, to'liqlik, baho, izoh

Kirish : clean_text (str), dars_nomi (str, ixtiyoriy)
Chiqish: str — ustozga hisobot
"""

import os

GEMINI_MODEL = "gemini-2.5-flash"   # bepul tier, eng tez


def analyze_report(clean_text: str, dars_nomi: str = "") -> str:
    """
    Gemini API orqali report tahlil qiladi.

    clean_text : postprocess.py dan kelgan tozalangan matn
    dars_nomi  : masalan "Dars 3 — Preprocessing" (ixtiyoriy)
    Returns    : ustozga o'zbekcha hisobot (str)
    """
    if not clean_text.strip():
        return "❌ Matn bo'sh — OCR hech narsa o'qiy olmadi. Rasmni tekshiring."

    api_key = _get_api_key()
    if api_key is None:
        return (
            "❌ Gemini API kalit topilmadi.\n"
            "Quyidagilardan birini bajaring:\n"
            "  1) .env fayliga:  GEMINI_API_KEY=sizning_kalitingiz\n"
            "  2) Terminal:      export GEMINI_API_KEY=sizning_kalitingiz\n"
            "Kalit olish (bepul): https://aistudio.google.com/app/apikey"
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return (
            "❌ 'google-genai' kutubxonasi o'rnatilmagan.\n"
            "Buyruq:  pip install -U google-genai"
        )

    client = genai.Client(api_key=api_key)

    dars_qism = f"Bu report '{dars_nomi}' darsi uchun yozilgan." if dars_nomi else ""

    prompt = f"""Sen tajribali o'qituvchi yordamchisisan. {dars_qism}
Quyidagi matn talaba qo'lda yozib, rasmga olib yuborgan DARS REPORTI.
OCR orqali o'qilgani uchun ba'zi xatolar bo'lishi mumkin — mazmunga e'tibor ber.

FAQAT quyidagi tuzilmada, o'zbek tilida (lotin) javob ber:

XULOSA       : (report nima haqida — 2-3 gap)
MAVZULAR     : (yoritilgan asosiy mavzular, ro'yxat ko'rinishida)
TO'LIQLIK    : (mavzu yetarli ochilganmi? — Yaxshi / O'rta / Yuzaki)
BAHO         : (10 ballik — masalan 8/10)
USTOZGA IZOH : (talabaga qanday maslahat berish kerak — 1-2 gap)

--- REPORT MATNI ---
{clean_text}
--- TUGADI ---"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini xatosi: {e}"


# ------------------------------------------------------------------
# Yordamchi
# ------------------------------------------------------------------
def _get_api_key() -> str | None:
    """Kalitni environment yoki .env fayldan oladi."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # .env faylni qo'lda o'qish (python-dotenv o'rnatilmagan bo'lsa ham)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return None
