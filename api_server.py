"""
api_server.py — SnapKnow's own REST API.

This exposes the same image-identification logic used by the Streamlit app
(app.py) as a standalone HTTP API, built with FastAPI. It reuses core.py so
there is exactly one implementation of the prompting/parsing logic behind
both the web UI and this API — they can never drift out of sync.

Access control
---------------
Every endpoint except / and /health requires a SnapKnow API key, sent in the
'X-API-Key' request header. This is separate from the Gemini/Claude/Ollama
key each caller also supplies in the request body — that one pays for their
own AI usage; this one controls who is allowed to call YOUR API at all.

To issue a key to someone:
    python manage_api_keys.py create "Their Name"

To list every key and how much each has been used:
    python manage_api_keys.py list

To revoke a key:
    python manage_api_keys.py revoke sk_xxxxxxxxxxxxxxxx

Keys are stored in api_keys.json next to this file. Add api_keys.json to
.gitignore — it should never be committed or shared publicly.

Run it with:
    uvicorn api_server:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger documentation
(FastAPI generates this automatically — a good thing to show in a demo). To
call a protected endpoint from Swagger, click "Authorize" and paste a key.
"""

import json
import os
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from core import (
    analyze_image,
    parse_response,
    resize_image_bytes,
    guess_media_type_from_bytes,
    translate_result_text,
)

app = FastAPI(
    title="SnapKnow API",
    description=(
        "Upload a photo and get it identified — category, name, description, "
        "details, and a fun fact — powered by Google Gemini or Anthropic Claude. "
        "Requires a SnapKnow API key in the 'X-API-Key' header — ask the project "
        "owner for one."
    ),
    version="1.0.0",
)

# Allow browser-based clients (e.g. a separate frontend) to call this API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SnapKnow API key store — a simple JSON file of {key: {owner, created,
# request_count, active}}. Managed with manage_api_keys.py (create/list/
# revoke), not through the API itself, so issuing a key is always a deliberate
# action by the project owner rather than something a caller can self-serve.
# ---------------------------------------------------------------------------
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.json")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> dict:
    if not os.path.exists(API_KEYS_FILE):
        return {}
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_api_keys(keys: dict):
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> dict:
    """FastAPI dependency: validates the 'X-API-Key' header against
    api_keys.json, tracks usage, and blocks the request if the key is
    missing, unrecognised, or has been revoked."""
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Send it in the 'X-API-Key' header. "
                   "Ask the project owner for one if you don't have one.",
        )
    keys = _load_api_keys()
    info = keys.get(key)
    if not info:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    if not info.get("active", True):
        raise HTTPException(status_code=403, detail="This API key has been revoked.")

    info["request_count"] = info.get("request_count", 0) + 1
    info["last_used"] = datetime.now().isoformat(timespec="seconds")
    keys[key] = info
    _save_api_keys(keys)
    return info


class AnalyzeResponse(BaseModel):
    caption: Optional[str] = None
    category: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None
    fun_fact: Optional[str] = None
    answer: Optional[str] = None


class TranslateRequest(BaseModel):
    result: dict
    target_language: str
    api_key: str
    model_name: str
    provider: str  # "gemini" or "claude"


def _to_response(parsed: dict) -> AnalyzeResponse:
    return AnalyzeResponse(
        caption=parsed.get("CAPTION"),
        category=parsed.get("CATEGORY"),
        name=parsed.get("NAME"),
        confidence=parsed.get("CONFIDENCE"),
        description=parsed.get("DESCRIPTION"),
        details=parsed.get("DETAILS"),
        fun_fact=parsed.get("FUN_FACT"),
        answer=parsed.get("ANSWER"),
    )


def _resolve_provider(provider: str) -> str:
    """Accept a few friendly spellings and map them to what core.analyze_image expects."""
    p = provider.strip().lower()
    if p in ("gemini", "google", "google gemini"):
        return "Google Gemini"
    if p in ("claude", "anthropic", "anthropic claude"):
        return "Anthropic Claude"
    if p in ("ollama", "local"):
        return "Ollama"
    raise HTTPException(status_code=400, detail="provider must be 'gemini', 'claude', or 'ollama'")


@app.get("/")
def root():
    return {
        "name": "SnapKnow API",
        "docs": "/docs",
        "auth": "Send your key in the 'X-API-Key' header on every endpoint below except / and /health.",
        "endpoints": ["/analyze", "/translate", "/keys/usage", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/keys/usage")
def key_usage(key_info: dict = Depends(require_api_key)):
    """Check your own key's usage — how many requests it's made and when it
    was last used. Requires a valid key, same as every other endpoint below;
    it only ever reports on the key you authenticated with, never anyone else's."""
    return {
        "owner": key_info.get("owner"),
        "created": key_info.get("created"),
        "last_used": key_info.get("last_used"),
        "request_count": key_info.get("request_count"),
    }


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/gif) to identify"),
    provider: str = Form(..., description="'gemini', 'claude', or 'ollama'"),
    model_name: str = Form(..., description="e.g. gemini-3.6-flash or claude-sonnet-4-6"),
    api_key: str = Form(..., description="API key for gemini/claude, or the Ollama base URL (e.g. http://localhost:11434) for ollama"),
    language: str = Form("English", description="Output language, e.g. English, Hindi, Marathi"),
    question: Optional[str] = Form(None, description="Optional follow-up question about the image"),
):
    """Identify the subject of an uploaded photo and return a structured result."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_bytes = resize_image_bytes(raw_bytes)
    media_type = guess_media_type_from_bytes(image_bytes)
    provider_name = _resolve_provider(provider)

    try:
        raw_text = analyze_image(
            image_bytes, media_type, api_key, model_name, provider_name,
            language=language, extra_question=question,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream AI provider error: {e}")

    parsed = parse_response(raw_text)
    if not parsed:
        raise HTTPException(status_code=502, detail="Could not parse a structured response from the model")

    return _to_response(parsed)


@app.post("/translate", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
def translate(req: TranslateRequest):
    """Translate an already-analyzed result into another language (text-only, no image needed)."""
    provider_name = _resolve_provider(req.provider)
    try:
        translated = translate_result_text(
            req.result, req.target_language, req.api_key, req.model_name, provider_name,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream AI provider error: {e}")
    return _to_response(translated)