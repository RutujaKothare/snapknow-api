"""
AI Image Caption & Info Assistant
----------------------------------
Upload an image (animal, flower, food, vegetable, or anything else) and this app will:
  1. Caption the image and identify it using Claude's vision (an LLM API).
  2. Give detailed information about it (habitat/diet for animals, nutrition for food, etc).
  3. Read the result aloud (Text-to-Speech).
  4. Let you ask follow-up questions by voice (Speech-to-Text).
  5. Let you download a PDF report of the analysis.

Run with:  streamlit run app.py
"""

import streamlit as st
import anthropic
from google import genai
from google.genai import types as genai_types
import base64
from PIL import Image
import io
from gtts import gTTS
import speech_recognition as sr
from fpdf import FPDF
import os
import json
import hashlib
import tempfile
from datetime import datetime

st.set_page_config(page_title="SnapKnow — AI Image Identifier", page_icon="📸", layout="wide")

# ---------------------------------------------------------------------------
# Visual theme — playful photo-scrapbook look. Results render as tilted
# polaroid cards with a sticker-style category tag and washi-tape corners;
# history becomes a small gallery of pinned thumbnails. Fits an app whose
# whole job is "you snapped a photo, here's what it is."
# ---------------------------------------------------------------------------
CATEGORY_STYLES = {
    "animal":    {"emoji": "🐾", "color": "#2F9E44"},
    "flower":    {"emoji": "🌸", "color": "#E64980"},
    "food":      {"emoji": "🍽️", "color": "#F76707"},
    "vegetable": {"emoji": "🥕", "color": "#74B816"},
    "fruit":     {"emoji": "🍎", "color": "#E03131"},
    "object":    {"emoji": "📦", "color": "#1971C2"},
    "place":     {"emoji": "🗺️", "color": "#7048E8"},
    "person":    {"emoji": "🧑", "color": "#0C8599"},
}
DEFAULT_STYLE = {"emoji": "✦", "color": "#495057"}

# ---------------------------------------------------------------------------
# Multi-language support — one code drives both the LLM prompt language and
# the gTTS voice used for "Read Aloud".
# ---------------------------------------------------------------------------
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Marathi (मराठी)": "mr",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
}

# ---------------------------------------------------------------------------
# Local history persistence — saves to a JSON file next to app.py so your
# analyzed images survive a browser refresh or restarting the app. Defined
# early since session-state init (below) needs it right away.
# ---------------------------------------------------------------------------
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapknow_history.json")


def load_history_from_disk() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            {"result": entry["result"], "image_bytes": base64.b64decode(entry["image_b64"])}
            for entry in raw
        ]
    except Exception:
        return []  # corrupted or unreadable file — start fresh rather than crashing


def save_history_to_disk(history: list):
    try:
        serializable = [
            {
                "result": entry["result"],
                "image_b64": base64.b64encode(entry["image_bytes"]).decode("ascii"),
            }
            for entry in history
        ]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception:
        pass  # best-effort — don't crash the app if the disk write fails


def category_style(category_text: str) -> dict:
    text = (category_text or "").lower()
    for key, style in CATEGORY_STYLES.items():
        if key in text:
            return style
    return DEFAULT_STYLE


HERO_SVG = """
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="30" width="100" height="72" rx="10" fill="#1F2937"/>
  <rect x="42" y="18" width="36" height="20" rx="5" fill="#1F2937"/>
  <circle cx="60" cy="68" r="24" fill="#F76707"/>
  <circle cx="60" cy="68" r="15" fill="#FFF3E0"/>
  <circle cx="88" cy="46" r="4" fill="#F2A93B"/>
</svg>
"""


def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg: #F5EFE3;
            --surface: #FFFFFF;
            --surface-alt: #FBF6EC;
            --ink: #1F2937;
            --ink-dim: #6B655A;
            --coral: #FF6B6A;
            --teal: #116466;
            --mustard: #F2A93B;
            --border: #E7DFCF;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg) !important;
            color: var(--ink) !important;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background-image: radial-gradient(rgba(31,41,55,0.05) 1.5px, transparent 1.5px);
            background-size: 22px 22px;
        }

        [data-testid="stHeader"] { background-color: transparent; }

        [data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 2px solid var(--ink);
        }
        [data-testid="stSidebar"] * { color: var(--ink) !important; }
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: var(--surface-alt) !important;
            border-radius: 8px !important;
            border: 1.5px solid var(--ink) !important;
        }

        h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: var(--ink) !important; font-weight: 700 !important; }

        [data-testid="stAppViewContainer"] .main .block-container { padding-top: 2.2rem; max-width: 1200px; }

        /* ---------- Hero: tilted postcard with tape corners ---------- */
        .hero-postcard {
            background-color: var(--surface);
            border: 2.5px solid var(--ink);
            border-radius: 14px;
            padding: 1.8rem 2.1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.8rem;
            transform: rotate(-0.6deg);
            box-shadow: 7px 7px 0 var(--ink);
            position: relative;
        }
        .hero-tape {
            position: absolute; top: -14px; width: 70px; height: 26px;
            background: rgba(242, 169, 59, 0.75);
            border: 1px solid rgba(31,41,55,0.15);
        }
        .hero-tape.left { left: 30px; transform: rotate(-8deg); }
        .hero-tape.right { right: 40px; transform: rotate(6deg); }
        .hero-text { max-width: 60ch; }
        .hero-eyebrow {
            font-family: 'Caveat', cursive;
            font-weight: 700;
            font-size: 1.6rem;
            color: var(--coral);
            margin-bottom: -0.3rem;
        }
        .hero-title {
            font-family: 'DM Sans', sans-serif;
            font-weight: 700;
            font-size: 2.4rem;
            letter-spacing: -0.02em;
            color: var(--ink);
            margin: 0 0 0.5rem 0;
        }
        .hero-sub { color: var(--ink-dim); font-size: 0.98rem; }

        [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
            background: var(--coral) !important;
            color: #ffffff !important;
            border: 2px solid var(--ink) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-family: 'DM Sans', sans-serif !important;
            box-shadow: 3px 3px 0 var(--ink);
            transition: transform 0.08s ease, box-shadow 0.08s ease;
        }
        [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
            transform: translate(-2px, -2px);
            box-shadow: 5px 5px 0 var(--ink);
        }
        [data-testid="stButton"] button:active, [data-testid="stDownloadButton"] button:active {
            transform: translate(0, 0);
            box-shadow: 1px 1px 0 var(--ink);
        }

        [data-testid="stTabs"] [role="tab"] {
            font-family: 'DM Sans', sans-serif;
            font-weight: 600;
            color: var(--ink-dim);
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--coral) !important;
            border-bottom: 3px solid var(--coral) !important;
        }

        [data-testid="stExpander"] {
            background-color: var(--surface) !important;
            border: 2px solid var(--ink) !important;
            border-radius: 12px !important;
            box-shadow: 4px 4px 0 rgba(31,41,55,0.15);
        }

        [data-testid="stFileUploaderDropzone"], [data-testid="stAudioInput"] {
            background-color: var(--surface-alt) !important;
            border: 2.5px dashed var(--ink) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"] *, [data-testid="stAudioInput"] *,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--ink) !important;
            font-family: 'DM Sans', sans-serif !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--ink-dim) !important; }
        [data-testid="stFileUploaderDropzone"] svg { fill: var(--coral) !important; opacity: 0.9; }
        [data-testid="stBaseButton-secondary"], [data-testid="stFileUploaderDropzone"] button {
            background: var(--teal) !important;
            color: #ffffff !important;
            border: 2px solid var(--ink) !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
        }

        /* Voice recorder — force the cream/paper background all the way down
           (Streamlit's default inner track renders near-black, which clashed
           with the rest of the theme) and keep icons clearly visible on it */
        [data-testid="stAudioInput"] {
            background-color: var(--surface-alt) !important;
            border: 2.5px dashed var(--ink) !important;
            border-radius: 12px !important;
        }
        [data-testid="stAudioInput"] div {
            background-color: var(--surface-alt) !important;
        }
        [data-testid="stAudioInput"] span, [data-testid="stAudioInput"] p {
            color: var(--ink) !important;
            opacity: 1 !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stAudioInput"] svg, [data-testid="stAudioInput"] path,
        [data-testid="stAudioInput"] circle {
            fill: var(--ink) !important;
            opacity: 0.7 !important;
        }
        [data-testid="stAudioInput"] canvas {
            filter: invert(0.75) contrast(1.3);
        }
        [data-testid="stAudioInput"] button {
            background: var(--coral) !important;
            border: 2px solid var(--ink) !important;
            border-radius: 50% !important;
        }
        [data-testid="stAudioInput"] button svg, [data-testid="stAudioInput"] button path {
            fill: #ffffff !important;
            opacity: 1 !important;
        }

        /* Collapsed-sidebar reopen button — make it a clearly visible coral
           circle instead of a faint default icon that's easy to lose track of */
        [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
            background-color: var(--coral) !important;
            border: 2px solid var(--ink) !important;
            border-radius: 50% !important;
            box-shadow: 2px 2px 0 var(--ink);
        }
        [data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg {
            fill: #ffffff !important;
        }

        [data-testid="stTextInput"] input {
            background-color: var(--surface) !important;
            color: var(--ink) !important;
            border: 2px solid var(--ink) !important;
            border-radius: 8px !important;
        }

        hr { border-top: 2px dashed var(--border) !important; }

        [data-testid="stCaptionContainer"], .stCaption { color: var(--ink-dim) !important; }

        [data-testid="stAlert"] {
            border-radius: 10px !important;
            background-color: #FFF3E0 !important;
            border: 2px solid var(--ink) !important;
        }
        [data-testid="stAlert"] * { color: var(--ink) !important; opacity: 1 !important; }

        /* ---------- Result card: tilted polaroid ---------- */
        .polaroid-card {
            background-color: var(--surface);
            border: 2px solid var(--ink);
            border-radius: 4px;
            padding: 1rem 1rem 1.6rem 1rem;
            margin-bottom: 1.2rem;
            transform: rotate(0.5deg);
            box-shadow: 6px 6px 0 rgba(31,41,55,0.18);
            position: relative;
        }
        .polaroid-sticker {
            position: absolute; top: -14px; left: 20px;
            font-family: 'DM Sans', sans-serif; font-weight: 700; font-size: 0.72rem;
            letter-spacing: 0.03em; text-transform: uppercase;
            color: #ffffff; padding: 0.3rem 0.7rem; border-radius: 999px;
            border: 2px solid var(--ink); transform: rotate(-3deg);
        }
        .polaroid-caption {
            font-family: 'Caveat', cursive; font-weight: 700; font-size: 1.7rem;
            color: var(--ink); margin: 0.6rem 0 0 0; line-height: 1.2;
        }
        .polaroid-name {
            color: var(--ink-dim); font-size: 0.85rem; margin-bottom: 0.4rem;
        }
        .polaroid-section-label {
            font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 700;
            color: var(--teal); text-transform: uppercase; letter-spacing: 0.05em;
            margin-top: 0.9rem; margin-bottom: 0.25rem;
        }
        .polaroid-body { color: var(--ink); line-height: 1.6; font-size: 0.95rem; }
        .polaroid-funfact {
            background-color: #FFF3E0;
            border: 1.5px dashed var(--mustard);
            border-radius: 8px; padding: 0.65rem 0.9rem; margin-top: 1rem;
            color: #8A5A00; font-family: 'Caveat', cursive; font-size: 1.15rem;
        }
        .polaroid-answer {
            background-color: #E6F7F7; border: 1.5px solid var(--teal);
            padding: 0.7rem 0.9rem; border-radius: 8px; margin-top: 1rem; color: var(--ink);
        }

        /* ---------- History gallery row ---------- */
        .history-badge {
            display: inline-block; font-size: 0.68rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.03em; color: #ffffff;
            padding: 0.18rem 0.55rem; border-radius: 999px; border: 1.5px solid var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

# ---------------------------------------------------------------------------
# Sidebar: provider + API key + model choice
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Settings")

provider = st.sidebar.radio(
    "AI Provider",
    ["Google Gemini (free tier)"],
    index=0,
    help="Gemini has a genuinely free tier (rate-limited, no card needed). Claude requires paid credits.",
)

if provider.startswith("Google"):
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey — sign in with a Google account, "
             "click 'Create API key'. No credit card required for the free tier.",
    )
    model_name = st.sidebar.selectbox(
        "Model",
        ["gemini-3.6-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="Flash and Flash-Lite are free-tier friendly (rate-limited). Pro Preview is paid-only "
             "but most capable — pick it only if you already have billing set up.",
    )


st.sidebar.markdown("---")
language_choice = st.sidebar.selectbox(
    "🌐 Output Language",
    list(LANGUAGE_OPTIONS.keys()),
    index=0,
    help="Captions, descriptions, details, fun facts, and read-aloud speech will all use this "
         "language. Changing this after analyzing a photo automatically re-explains it in the "
         "new language.",
)
tts_lang = LANGUAGE_OPTIONS[language_choice]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Image captioning & knowledge run on your chosen provider above. Text-to-speech uses gTTS "
    "and speech-to-text uses Google's free web speech service — both work regardless of which "
    "provider you pick. History is saved locally next to app.py so it survives a refresh."
)

st.markdown(
    f"""
    <div class="hero-postcard">
        <div class="hero-tape left"></div>
        <div class="hero-tape right"></div>
        <div class="hero-text">
            <div class="hero-eyebrow">Snap it. Know it.</div>
            <div class="hero-title">📸 SnapKnow</div>
            <div class="hero-sub">
                Upload any photo — an animal, a flower, a dish, anything — and get it identified,
                explained, read aloud, and saved as a PDF report.
            </div>
        </div>
        <div>{HERO_SVG}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_disk()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_language" not in st.session_state:
    st.session_state.last_language = language_choice
if "translation_cache" not in st.session_state:
    # Caches {(image_hash, language): parsed_result} so switching back to a
    # language you've already viewed for this image is instant, no API call.
    st.session_state.translation_cache = {}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_media_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/jpeg")


def resize_image_bytes(image_bytes: bytes, max_dimension: int = 1280) -> bytes:
    """Downscale large photos before sending to the API. This is what actually makes
    analysis and follow-up questions noticeably faster — a smaller payload uploads
    faster and the vision model processes it faster too. Phone photos are often
    3000-4000px wide; the model doesn't need that much detail to identify a subject."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_format = (img.format or "JPEG").upper()
        width, height = img.size
        if max(width, height) <= max_dimension:
            return image_bytes  # already small enough
        scale = max_dimension / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        is_jpeg = img_format in ("JPEG", "JPG")
        if is_jpeg:
            img = img.convert("RGB")
        img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        save_kwargs = {"quality": 85} if is_jpeg else {}
        img.save(buf, format="JPEG" if is_jpeg else img_format, **save_kwargs)
        return buf.getvalue()
    except Exception:
        return image_bytes  # if anything goes wrong, fall back to the original


def guess_media_type_from_bytes(image_bytes: bytes) -> str:
    """Figure out the mime type when we only have raw bytes (e.g. re-analyzing an
    already-loaded image after a language switch, with no filename available)."""
    try:
        fmt = (Image.open(io.BytesIO(image_bytes)).format or "JPEG").upper()
        return {
            "JPEG": "image/jpeg", "JPG": "image/jpeg", "PNG": "image/png",
            "WEBP": "image/webp", "GIF": "image/gif",
        }.get(fmt, "image/jpeg")
    except Exception:
        return "image/jpeg"


def image_hash(image_bytes: bytes) -> str:
    """Short fingerprint used as a cache key so re-analyzing the same image in a
    language you've already viewed doesn't need a fresh API call."""
    return hashlib.md5(image_bytes).hexdigest()


ANALYSIS_PROMPT = """You are an expert naturalist, botanist, chef, and general knowledge assistant.
Look at this image carefully and respond ONLY in the following structured format (keep the field
names exactly as shown, one field per line, content can span multiple lines):

CAPTION: <one-line caption describing the image>
CATEGORY: <Animal / Flower / Food / Vegetable / Fruit / Object / Place / Person / Other>
NAME: <specific name/species/dish name if identifiable>
DESCRIPTION: <2-4 sentences describing what is seen>
DETAILS: <detailed facts - for animals: habitat, diet, lifespan, behavior; for plants/flowers: species, growing conditions, symbolism; for food/vegetables: nutrition, origin, how it's used/cooked; for anything else: relevant interesting facts>
FUN_FACT: <one interesting/fun fact>
"""


def analyze_image(image_bytes, media_type, api_key, model_name, provider, language="English", extra_question=None):
    """Route to the selected provider's vision API and get a structured description back."""
    prompt = ANALYSIS_PROMPT
    if language != "English":
        prompt += (
            f"\n\nWrite the CAPTION, DESCRIPTION, DETAILS, and FUN_FACT fields in {language}. "
            f"Keep the NAME field in English (the common/scientific name), since it's used for "
            f"look-ups elsewhere. Keep the field labels themselves (CAPTION:, CATEGORY:, etc.) in English."
        )
    if extra_question:
        answer_lang_note = f" (write the answer in {language})" if language != "English" else ""
        prompt += (
            f"\n\nAlso specifically answer this follow-up question about the image: "
            f"{extra_question}\nPut your answer on its own line as:\nANSWER: <your answer here>{answer_lang_note}"
        )

    if provider.startswith("Google"):
        return _analyze_with_gemini(image_bytes, media_type, api_key, model_name, prompt)
    else:
        return _analyze_with_claude(image_bytes, media_type, api_key, model_name, prompt)


def _analyze_with_claude(image_bytes, media_type, api_key, model_name, prompt):
    client = anthropic.Anthropic(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return message.content[0].text


def _analyze_with_gemini(image_bytes, media_type, api_key, model_name, prompt):
    client = genai.Client(api_key=api_key)
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, image_part],
    )
    return response.text


def translate_result_text(result: dict, target_language: str, api_key: str, model_name: str, provider: str) -> dict:
    """Translate an already-generated result's text fields into another language using
    a fast TEXT-ONLY API call — no image re-upload needed, so this is much quicker than
    re-running the full vision analysis just to switch languages. NAME is left untouched
    since it's kept in English throughout for consistency."""
    translatable = {
        k: v for k, v in result.items()
        if k in ("CAPTION", "DESCRIPTION", "DETAILS", "FUN_FACT", "ANSWER") and v
    }
    if not translatable:
        return result

    prompt = (
        f"Translate the text after each label below into {target_language}. Reply with ONLY "
        f"the same labels, one per line, in the exact format 'LABEL: translated text'. Do not "
        f"translate the labels themselves, and do not add any commentary.\n\n"
    )
    for key, value in translatable.items():
        prompt += f"{key}: {value}\n"

    if provider.startswith("Google"):
        raw = _translate_with_gemini(prompt, api_key, model_name)
    else:
        raw = _translate_with_claude(prompt, api_key, model_name)

    translated_fields = parse_response(raw)
    new_result = dict(result)
    for key in translatable:
        if translated_fields.get(key):
            new_result[key] = translated_fields[key]
    return new_result


def _translate_with_claude(prompt, api_key, model_name):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _translate_with_gemini(prompt, api_key, model_name):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=[prompt])
    return response.text


def parse_response(text: str) -> dict:
    """Turn the structured LLM text into a dict of fields."""
    fields = ["CAPTION", "CATEGORY", "NAME", "DESCRIPTION", "DETAILS", "FUN_FACT", "ANSWER"]
    result = {}
    current_field = None
    buffer = []
    for line in text.split("\n"):
        matched = False
        for f in fields:
            if line.strip().startswith(f + ":"):
                if current_field:
                    result[current_field] = "\n".join(buffer).strip()
                current_field = f
                buffer = [line.split(":", 1)[1].strip()]
                matched = True
                break
        if not matched and current_field:
            buffer.append(line)
    if current_field:
        result[current_field] = "\n".join(buffer).strip()
    return result


def text_to_speech(text: str, lang: str = "en") -> str:
    """Generate an mp3 file from text and return its path."""
    tts = gTTS(text=text, lang=lang)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name


def speech_to_text(audio_bytes: bytes):
    """Transcribe recorded audio bytes (wav) to text using free Google Web Speech API."""
    recognizer = sr.Recognizer()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.write(audio_bytes)
    tmp_file.close()
    try:
        with sr.AudioFile(tmp_file.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


def generate_pdf(image_bytes: bytes, parsed: dict, filename: str = "image_report.pdf") -> str:
    """Build a nicely formatted PDF report with the image + all extracted info."""
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_img.close()  # release the handle immediately — Windows can't delete an open file later otherwise
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(tmp_img.name, "JPEG")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI Image Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 8, datetime.now().strftime("Generated on %Y-%m-%d %H:%M"),
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(4)

    pdf.image(tmp_img.name, x=55, w=100)
    # pdf.image() can leave the cursor at the image's right edge; explicitly
    # reset it to the left margin below the image before writing more text,
    # otherwise multi_cell() has no horizontal room left and raises
    # "Not enough horizontal space to render a single character". Compute the
    # exact rendered height from the image's own aspect ratio (width is fixed
    # at 100mm above) so the cursor lands right below the image, not on it.
    img_w_px, img_h_px = img.size
    rendered_height_mm = 100 * (img_h_px / img_w_px)
    new_y = pdf.get_y() + rendered_height_mm + 4
    if new_y > pdf.page_break_trigger:
        pdf.add_page()
        new_y = pdf.get_y()
    pdf.set_xy(pdf.l_margin, new_y)

    def safe(text: str) -> str:
        # fpdf2's built-in Helvetica font only supports Latin-1; replace any
        # characters outside that range (smart quotes, emoji, etc.) so the
        # PDF doesn't crash or render garbled boxes.
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 10, safe(parsed.get("CAPTION", "")))

    for field, label in [
        ("CATEGORY", "Category"),
        ("NAME", "Name"),
        ("DESCRIPTION", "Description"),
        ("DETAILS", "Details"),
        ("FUN_FACT", "Fun Fact"),
    ]:
        if parsed.get(field):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, label + ":")
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, safe(parsed[field]))
            pdf.ln(2)

    out_path = os.path.join(tempfile.gettempdir(), filename)
    pdf.output(out_path)
    try:
        os.unlink(tmp_img.name)
    except OSError:
        pass  # best-effort cleanup; a lingering temp file isn't worth crashing the app over
    return out_path


def generate_combined_pdf(history: list, filename: str = "snapknow_all_history.pdf") -> str:
    """Build one PDF containing every analyzed image + its info, one entry per page."""

    def safe(text: str) -> str:
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for entry in history:
        parsed = entry["result"]
        image_bytes = entry["image_bytes"]

        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp_img.close()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(tmp_img.name, "JPEG")

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, safe(parsed.get("CAPTION", "")))
        pdf.image(tmp_img.name, x=55, w=100)

        img_w_px, img_h_px = img.size
        rendered_height_mm = 100 * (img_h_px / img_w_px)
        new_y = pdf.get_y() + rendered_height_mm + 4
        if new_y > pdf.page_break_trigger:
            pdf.add_page()
            new_y = pdf.get_y()
        pdf.set_xy(pdf.l_margin, new_y)

        for field, label in [
            ("CATEGORY", "Category"),
            ("NAME", "Name"),
            ("DESCRIPTION", "Description"),
            ("DETAILS", "Details"),
            ("FUN_FACT", "Fun Fact"),
        ]:
            if parsed.get(field):
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 8, label + ":")
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 7, safe(parsed[field]))
                pdf.ln(2)

        try:
            os.unlink(tmp_img.name)
        except OSError:
            pass

    out_path = os.path.join(tempfile.gettempdir(), filename)
    pdf.output(out_path)
    return out_path


# ---------------------------------------------------------------------------
# If the person changes the language dropdown after already having a result
# on screen, translate the existing text into the new language. Cache hits
# apply instantly here; anything needing a real API call is deferred to a
# flag checked inside col2, so the rest of the page (upload box, history)
# keeps rendering normally instead of the whole app going blank while we wait.
# ---------------------------------------------------------------------------
if "pending_translation" not in st.session_state:
    st.session_state.pending_translation = False

if (
    language_choice != st.session_state.last_language
    and st.session_state.last_image_bytes is not None
    and st.session_state.last_result is not None
):
    st.session_state.last_language = language_choice
    cache_key = (image_hash(st.session_state.last_image_bytes), language_choice)
    cached = st.session_state.translation_cache.get(cache_key)
    if cached:
        st.session_state.last_result = cached
        st.session_state.qa_history = []
        st.session_state.pending_translation = False
    elif api_key:
        st.session_state.pending_translation = True
    else:
        st.session_state.pending_translation = False
        st.warning("Enter your API key in the sidebar to switch languages on an existing result.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"file_uploader_{st.session_state.uploader_key}",
    )
    if uploaded_file:
        image_bytes = resize_image_bytes(uploaded_file.getvalue())
        st.session_state.last_image_bytes = image_bytes
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
            else:
                with st.spinner(f"Analyzing image with {provider.split(' (')[0]}..."):
                    media_type = get_media_type(uploaded_file.name)
                    try:
                        raw = analyze_image(
                            image_bytes, media_type, api_key, model_name, provider, language_choice
                        )
                        parsed = parse_response(raw)
                        st.session_state.last_result = parsed
                        st.session_state.qa_history = []
                        st.session_state.translation_cache[
                            (image_hash(image_bytes), language_choice)
                        ] = parsed
                        st.session_state.history.append(
                            {"result": parsed, "image_bytes": image_bytes}
                        )
                        save_history_to_disk(st.session_state.history)
                    except Exception as e:
                        st.error(f"Error calling API: {e}")

    elif st.session_state.last_image_bytes:
        # No fresh upload this run (e.g. the user picked an image from History),
        # but we still have an image loaded — show it so it isn't blank.
        st.image(st.session_state.last_image_bytes, caption="Selected image", use_container_width=True)
        st.caption("Loaded from history. Upload a new file above to analyze a different image.")

    st.divider()
    st.subheader("💬 Ask a follow-up question")

    text_tab, voice_tab = st.tabs(["⌨️ Type", "🎤 Voice"])

    with text_tab:
        text_question = st.text_input(
            "Type your question about the image",
            placeholder="e.g. Is this safe for dogs to eat?",
            key="text_question_input",
        )
        if st.button("➡️ Ask", key="ask_text_button"):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
            elif st.session_state.last_image_bytes is None:
                st.error("Upload and analyze an image first.")
            elif not text_question.strip():
                st.error("Type a question first.")
            else:
                with st.spinner("Getting answer..."):
                    media_type = get_media_type("img.jpg")
                    raw = analyze_image(
                        st.session_state.last_image_bytes,
                        media_type,
                        api_key,
                        model_name,
                        provider,
                        language_choice,
                        extra_question=text_question.strip(),
                    )
                    parsed = parse_response(raw)
                    st.session_state.last_result = parsed
                    st.session_state.qa_history.append(
                        {"question": text_question.strip(), "answer": parsed.get("ANSWER", "")}
                    )

    with voice_tab:
        st.caption("Click the mic, allow browser microphone access, record your question, then click below.")
        audio_value = st.audio_input("Record your question")
        if audio_value and st.button("🗣️ Transcribe & Ask", key="ask_voice_button"):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
            elif st.session_state.last_image_bytes is None:
                st.error("Upload and analyze an image first.")
            else:
                with st.spinner("Transcribing..."):
                    question = speech_to_text(audio_value.getvalue())
                if not question:
                    st.error("Could not understand the audio, please try again.")
                elif isinstance(question, str) and question.startswith("ERROR:"):
                    st.error(question)
                else:
                    st.success(f"You asked: {question}")
                    with st.spinner("Getting answer..."):
                        media_type = get_media_type("img.jpg")
                        raw = analyze_image(
                            st.session_state.last_image_bytes,
                            media_type,
                            api_key,
                            model_name,
                            provider,
                            language_choice,
                            extra_question=question,
                        )
                        parsed = parse_response(raw)
                        st.session_state.last_result = parsed
                        st.session_state.qa_history.append(
                            {"question": question, "answer": parsed.get("ANSWER", "")}
                        )

with col2:
    if st.session_state.pending_translation:
        with st.spinner(f"Translating to {language_choice}..."):
            try:
                translated = translate_result_text(
                    st.session_state.last_result, language_choice, api_key, model_name, provider
                )
                cache_key = (image_hash(st.session_state.last_image_bytes), language_choice)
                st.session_state.last_result = translated
                st.session_state.translation_cache[cache_key] = translated
                st.session_state.qa_history = []
            except Exception as e:
                st.error(f"Could not translate to {language_choice}: {e}")
            finally:
                st.session_state.pending_translation = False

    result = st.session_state.last_result
    if result:
        style = category_style(result.get("CATEGORY", ""))
        card_html = f"""
        <div class="polaroid-card">
            <span class="polaroid-sticker" style="background-color:{style['color']};">
                {style['emoji']} {result.get('CATEGORY', 'Unclassified')}
            </span>
            <div class="polaroid-caption">{result.get('CAPTION', '')}</div>
            <div class="polaroid-name">{result.get('NAME', '')}</div>
            <div class="polaroid-section-label">Description</div>
            <div class="polaroid-body">{result.get('DESCRIPTION', '')}</div>
            <div class="polaroid-section-label">Details</div>
            <div class="polaroid-body">{result.get('DETAILS', '')}</div>
            <div class="polaroid-funfact">✨ {result.get('FUN_FACT', '')}</div>
            {f'<div class="polaroid-answer"><strong>💬 Answer:</strong> {result["ANSWER"]}</div>' if result.get('ANSWER') else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        if st.session_state.qa_history:
            with st.expander(f"💬 Follow-up Q&A ({len(st.session_state.qa_history)})"):
                for qa in reversed(st.session_state.qa_history):
                    st.markdown(f"**Q:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']}")
                    st.markdown("---")

        full_text = " ".join(
            [result.get(f, "") for f in ["CAPTION", "DESCRIPTION", "DETAILS", "FUN_FACT"]]
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("🔊 Read Aloud"):
                with st.spinner("Generating speech..."):
                    audio_path = text_to_speech(full_text, lang=tts_lang)
                st.audio(audio_path, format="audio/mp3")
        with colB:
            if st.session_state.last_image_bytes:
                pdf_path = generate_pdf(st.session_state.last_image_bytes, result)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download PDF Report",
                        f,
                        file_name="image_report.pdf",
                        mime="application/pdf",
                    )
    else:
        st.info("Upload an image and click **Analyze Image** to see results here.")

st.divider()
with st.expander("🗂️ History of analyzed images", expanded=False):
    if not st.session_state.history:
        st.caption("No images analyzed yet.")
    else:
        hdr_col1, hdr_col2 = st.columns(2)
        with hdr_col1:
            combined_pdf_path = generate_combined_pdf(st.session_state.history)
            with open(combined_pdf_path, "rb") as f:
                st.download_button(
                    "📚 Export All as PDF",
                    f,
                    file_name="snapknow_all_history.pdf",
                    mime="application/pdf",
                    key="export_all_history",
                )
        with hdr_col2:
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.history = []
                save_history_to_disk(st.session_state.history)
                st.rerun()
        st.markdown("---")

    for i, h in enumerate(reversed(st.session_state.history)):
        entry_result = h["result"]
        entry_image = h["image_bytes"]
        entry_style = category_style(entry_result.get("CATEGORY", ""))
        hcol1, hcol2, hcol3 = st.columns([1, 4, 1.3])
        with hcol1:
            st.image(entry_image, width=60)
        with hcol2:
            st.markdown(f"**{entry_result.get('CAPTION', '')}**")
            st.markdown(
                f"<span class='history-badge' style='background-color:{entry_style['color']};'>"
                f"{entry_style['emoji']} {entry_result.get('CATEGORY', '')}</span>",
                unsafe_allow_html=True,
            )
        with hcol3:
            if st.button("🔎 View", key=f"view_history_{i}"):
                st.session_state.last_result = entry_result
                st.session_state.last_image_bytes = entry_image
                st.session_state.qa_history = []
                # Bump the uploader's key so the file_uploader widget resets to
                # empty on rerun — otherwise it keeps re-displaying whatever was
                # uploaded earlier instead of the historical image just selected.
                st.session_state.uploader_key += 1
                st.rerun()
        st.markdown("---")