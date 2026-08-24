# 🖼️ AI Image Caption & Info Assistant

An AI-powered web app that:

1. **Image Captioning + Knowledge** — upload a photo (animal, flower, food, vegetable, object, anything)
   and Gemini or Claude's vision model captions it and gives you a full write-up: category, name/species,
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
7. **Wikipedia lookup** — a verified summary and source link for whatever was identified (no API key needed).
8. **USDA nutrition facts** — real calorie/protein/carb data for Food, Vegetable, and Fruit results
   (free key from fdc.nal.usda.gov/api-key-signup, or the shared DEMO_KEY for light testing).
9. **GBIF species data** — scientific name, taxonomy, and conservation status for Animal and Flower
   results (no API key needed).
10. **Unsplash reference photos** — 2-3 real photos of the identified subject, for a visual sanity check
    (free key from unsplash.com/developers).
11. **QR code in PDF reports** — every downloaded PDF includes a scannable QR code linking to a Wikipedia
    search for the identified subject (no API key needed).
12. **Recipes for Food results** — a full ingredient list and cooking instructions from TheMealDB
    (no API key needed).
13. **Weather for Place results** — current conditions and temperature for identified places/landmarks
    (free key from openweathermap.org).

## Extra API keys (optional)

Wikipedia, GBIF, TheMealDB, and the QR code generator work immediately with no signup. USDA,
Unsplash, and OpenWeatherMap are optional — leave their sidebar fields blank to skip those features,
or grab free keys here:

- USDA FoodData Central: https://fdc.nal.usda.gov/api-key-signup
- Unsplash: https://unsplash.com/developers
- OpenWeatherMap: https://openweathermap.org/api

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

### Option B — Anthropic Claude (paid, more accurate on some tasks)

1. Go to https://console.anthropic.com/
2. Create an account / sign in, add a payment method, then create an API key.
3. There's no free tier for the Claude API — you'll need at least a small amount of credit.

Either way, you'll paste the key into the app's sidebar when you run it — it is only kept in
your browser session and never stored on disk or sent anywhere except that provider's API.

## 3. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## 4. How to use it

1. Paste your Anthropic API key into the sidebar and pick a model (Sonnet is a good default).
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
├── app.py              # Streamlit web app (UI, voice, PDF, history)
├── core.py             # Shared image-analysis logic (used by both app.py and api_server.py)
├── api_server.py        # Standalone FastAPI REST API — SnapKnow's own API
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

`app.py` and `api_server.py` both import their core logic (prompting, response
parsing, image resizing, translation) from `core.py`, so there is exactly one
implementation behind the web UI and the API — they can't drift out of sync.

## 5a. Running your own API (api_server.py)

On top of the Streamlit app, this project also exposes its own REST API, built
with FastAPI, so other programs (or a professor testing it independently) can
call SnapKnow's image analysis directly over HTTP.

```bash
uvicorn api_server:app --reload
```

Then open **http://127.0.0.1:8000/docs** — FastAPI auto-generates interactive
Swagger documentation where you can try the API directly from the browser.

**Endpoints:**

- `POST /analyze` — upload an image (`file`) plus `provider` (`gemini` or `claude`),
  `model_name`, `api_key`, and optionally `language` and `question`. Returns the
  parsed result (caption, category, name, description, details, fun fact) as JSON.
- `POST /translate` — send an already-analyzed `result` dict plus a `target_language`
  to translate it, without re-uploading the image.
- `GET /health` — simple health check, returns `{"status": "ok"}`.

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@/path/to/photo.jpg" \
  -F "provider=gemini" \
  -F "model_name=gemini-3.6-flash" \
  -F "api_key=YOUR_GEMINI_API_KEY"
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