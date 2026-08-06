# 🖼️ AI Image Caption & Info Assistant

An AI-powered web app that:

1. **Image Captioning + Knowledge** — upload a photo (animal, flower, food, vegetable, object, anything)
   and Gemini vision model captions it and gives you a full write-up: category, name/species,
   description, detailed facts (habitat/diet for animals, nutrition/origin for food, etc.), and a fun fact.
2. **Text-to-Speech** — click "Read Aloud" to have the result spoken back to you, in your chosen language.
3. **Speech-to-Text** — type or record a follow-up voice question about the image and get it answered.
4. **PDF Report** — download a formatted PDF containing the image and all the extracted information,
   or export your entire history as one combined PDF.
5. **Multi-language output** — pick English, Hindi, Marathi, Spanish, or French in the sidebar; captions,
   descriptions, fun facts, and read-aloud speech all switch to that language. Changing the language
   after you've already analyzed a photo automatically re-explains that same photo in the new language.
6. **Persistent history** — every analyzed image is saved to a local `snapknow_history.json` file next
   to `app.py`, so your history survives closing the browser tab or restarting the app.

## 1. Setup

```bash
# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on audio:** `gTTS` and `SpeechRecognition` need an internet connection (they call free
> Google web services). No extra API key is needed for those two. Speech-to-text uses your
> browser's microphone through Streamlit's built-in `st.audio_input` widget (Streamlit ≥ 1.38),
> so make sure you allow microphone access when prompted.

## 2. Get an API key

The app supports two providers — pick one in the sidebar.

### Option A — Google Gemini (free tier, no card required) — recommended to start

1. Go to https://aistudio.google.com/apikey
2. Sign in with any Google account and click **Create API key**.
3. Copy the key — no billing setup needed to use the free tier.
4. Free tier is rate-limited (requests per minute/day) rather than unlimited, and the exact
   numbers change over time — check the limits shown in AI Studio if you hit a `429` error.
   Google also retires older model IDs over time (e.g. `gemini-2.5-flash-lite` is closed to new
   users) — if you get a `404 NOT_FOUND "model ... no longer available"` error, open
   https://ai.google.dev/gemini-api/docs/models to see the current lineup and update the model
   list in `app.py`. As of August 2026, `gemini-3.6-flash` and `gemini-3.5-flash-lite` are the
   current free-tier-friendly models; `gemini-3.1-pro-preview` is paid-only.

## 3. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## 4. How to use it

1. Paste your Gemini API key into the sidebar and pick a model (Sonnet is a good default).
2. Upload a photo using the file uploader.
3. Click **🔍 Analyze Image** — Claude will look at the picture and return:
   - Caption
   - Category (Animal / Flower / Food / Vegetable / Fruit / Object / Place / Person / Other)
   - Name / species / dish name
   - Description
   - Detailed facts
   - A fun fact
4. Click **🔊 Read Aloud** to hear the result as speech.
5. Use the microphone recorder to ask a follow-up question ("Is this safe for dogs to eat?",
   "Where is this flower usually found?", etc.) and click **🗣️ Transcribe & Ask**.
6. Click **📄 Download PDF Report** to save everything as a PDF.

## 5. Project structure

```
image_ai_project/
├── app.py              # Main Streamlit application (all logic lives here)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 6. Ideas to extend this project

- **Multiple images at once** — loop the analysis over a batch of uploaded files and produce one
  combined PDF.
- **Nutrition database lookup** — for food/vegetables, cross-check with a nutrition API
  (e.g. USDA FoodData Central) for exact calorie/macro data instead of relying purely on the LLM.
- **More languages** — add more entries to the `LANGUAGE_OPTIONS` dict in `app.py` (any gTTS-supported
  language code works) if English/Hindi/Marathi/Spanish/French isn't enough.
- **Move history to a real database** — swap the local `snapknow_history.json` file for SQLite if you
  want multi-user support or want to query/filter history more richly.
- **Deploy** — this app deploys as-is to Streamlit Community Cloud, Hugging Face Spaces, or any
  Docker host. Just remember to let each user supply their own API key rather than hard-coding one.
- **Swap the LLM provider** — the `analyze_image()` function is intentionally isolated, so you can
  swap in OpenAI's GPT-4o vision or Google Gemini vision by changing just that one function.

## 7. Troubleshooting

- **"Error calling API"** — double check the API key is correct and has available credits.
- **Microphone recorder not appearing** — upgrade Streamlit: `pip install -U streamlit`.
- **`recognize_google` errors / no internet** — speech-to-text needs internet access since it
  calls a free Google web service; for a fully offline alternative, swap in `openai-whisper`
  (`pip install openai-whisper`) and transcribe locally.
- **PDF has garbled special characters** — `fpdf2`'s built-in fonts only support Latin-1 text.
  For a response containing non-Latin characters, add a Unicode TTF font with
  `pdf.add_font(...)` before calling `pdf.set_font(...)`.