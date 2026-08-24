"""
core.py — shared image-analysis logic for SnapKnow.

This module has NO Streamlit dependency on purpose: it's imported by both
app.py (the Streamlit UI) and api_server.py (the standalone FastAPI service),
so the exact same prompting, parsing, and resizing logic powers both — no
duplicated/drifting copies of the same code.
"""

import base64
import io
import logging
from PIL import Image
import anthropic
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger("snapknow.core")


def get_media_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/jpeg")


def guess_media_type_from_bytes(image_bytes: bytes) -> str:
    """Figure out the mime type when we only have raw bytes (no filename)."""
    try:
        fmt = (Image.open(io.BytesIO(image_bytes)).format or "JPEG").upper()
        return {
            "JPEG": "image/jpeg", "JPG": "image/jpeg", "PNG": "image/png",
            "WEBP": "image/webp", "GIF": "image/gif",
        }.get(fmt, "image/jpeg")
    except Exception:
        logger.warning("Could not determine image format from bytes; defaulting to image/jpeg", exc_info=True)
        return "image/jpeg"


def resize_image_bytes(image_bytes: bytes, max_dimension: int = 1280) -> bytes:
    """Downscale large photos before sending to the API — smaller payload,
    faster upload, faster processing. Phone photos are often 3000-4000px wide;
    the model doesn't need that much detail to identify a subject."""
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
        logger.warning("Image resize failed; sending original image unresized", exc_info=True)
        return image_bytes  # if anything goes wrong, fall back to the original


ANALYSIS_PROMPT = """You are an expert naturalist, botanist, chef, and general knowledge assistant.
Look at this image carefully and respond ONLY in the following structured format (keep the field
names exactly as shown, one field per line, content can span multiple lines):

CAPTION: <one-line caption describing the image>
CATEGORY: <Animal / Flower / Food / Vegetable / Fruit / Object / Place / Person / Other>
NAME: <specific name/species/dish name if identifiable>
CONFIDENCE: <a single whole number 0-100, your honest confidence that this identification is correct>
DESCRIPTION: <2-4 sentences describing what is seen>
DETAILS: <detailed facts - for animals: habitat, diet, lifespan, behavior; for plants/flowers: species, growing conditions, symbolism; for food/vegetables: nutrition, origin, how it's used/cooked; for anything else: relevant interesting facts>
FUN_FACT: <one interesting/fun fact>
"""


def analyze_image(image_bytes, media_type, api_key, model_name, provider, language="English", extra_question=None):
    """Route to the selected provider's vision API and get a structured description back.
    For Ollama, `api_key` is repurposed as the Ollama base URL (e.g. http://localhost:11434) —
    Ollama runs locally and doesn't use API keys."""
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
    elif provider.startswith("Ollama"):
        return _analyze_with_ollama(image_bytes, media_type, api_key, model_name, prompt)
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


def _analyze_with_ollama(image_bytes, media_type, base_url, model_name, prompt):
    """Runs entirely on your own machine via Ollama (https://ollama.com) — no API key,
    no cloud provider, no per-request cost. Needs a vision-capable model pulled locally
    first, e.g.: `ollama pull llava` or `ollama pull llama3.2-vision`."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatOllama(base_url=base_url or "http://localhost:11434", model=model_name, temperature=0.3)
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:{media_type};base64,{b64}"},
    ])
    response = llm.invoke([message])
    return response.content


def translate_result_text(result: dict, target_language: str, api_key: str, model_name: str, provider: str) -> dict:
    """Translate an already-generated result's text fields into another language using
    a fast TEXT-ONLY API call — no image re-upload needed. NAME is left untouched since
    it's kept in English throughout for consistency."""
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

    raw = text_completion(prompt, api_key, model_name, provider)

    translated_fields = parse_response(raw)
    new_result = dict(result)
    for key in translatable:
        if translated_fields.get(key):
            new_result[key] = translated_fields[key]
    return new_result


def text_completion(prompt: str, api_key: str, model_name: str, provider: str) -> str:
    """Generic text-only completion — one prompt in, one text reply out. Used for
    translation, and also as an AI fallback when a real external API (Wikipedia,
    USDA, GBIF, etc.) has no data for a given item."""
    if provider.startswith("Google"):
        return _complete_with_gemini(prompt, api_key, model_name)
    elif provider.startswith("Ollama"):
        return _complete_with_ollama(prompt, api_key, model_name)
    else:
        return _complete_with_claude(prompt, api_key, model_name)


def _complete_with_claude(prompt, api_key, model_name):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _complete_with_gemini(prompt, api_key, model_name):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=[prompt])
    return response.text


def _complete_with_ollama(prompt, base_url, model_name):
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    llm = ChatOllama(base_url=base_url or "http://localhost:11434", model=model_name, temperature=0.3)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def parse_response(text: str) -> dict:
    """Turn the structured LLM text into a dict of fields."""
    fields = ["CAPTION", "CATEGORY", "NAME", "CONFIDENCE", "DESCRIPTION", "DETAILS", "FUN_FACT", "ANSWER"]
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