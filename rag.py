"""
rag.py — Retrieval-Augmented Generation layer for SnapKnow.

SnapKnow already calls several *verified* data sources for every photo it
analyzes: Wikipedia, USDA FoodData Central, GBIF, TheMealDB, and
OpenWeatherMap. This module turns those lookups into a small, persistent,
searchable knowledge base, and provides simple TF-IDF retrieval over it.

Why TF-IDF instead of embeddings?
    - No extra API key, no model download, no network call to retrieve.
    - Fast and fully deterministic for a knowledge base this size
      (dozens-hundreds of short fact snippets, not millions of documents).
    - Easy to swap for a real embedding index later if the knowledge base
      grows large — the load/save/retrieve interface below would stay
      the same, only `_similarity_scores` would change.

No Streamlit dependency on purpose, mirroring core.py, so this can be
reused from api_server.py too if needed.
"""

import os
import json
import time
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("snapknow.rag")

KB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapknow_knowledge_base.json")

# Facts scoring below this similarity to the query are treated as "not
# relevant" and dropped, rather than forcing in a weak/unrelated match.
RELEVANCE_THRESHOLD = 0.08


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load_kb() -> list:
    if not os.path.exists(KB_FILE):
        return []
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.error(f"Could not load knowledge base from {KB_FILE}; starting empty", exc_info=True)
        return []


def save_kb(docs: list):
    try:
        with open(KB_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.error(f"Could not save knowledge base to {KB_FILE}", exc_info=True)


def _make_doc(image_name: str, category: str, source: str, text: str) -> dict:
    return {
        "id": f"{source}:{image_name}:{int(time.time() * 1000)}",
        "image_name": image_name,
        "category": category,
        "source": source,
        "text": text.strip(),
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def add_enrichment_to_kb(
    name: str,
    category: str,
    wiki: dict = None,
    usda: dict = None,
    gbif: dict = None,
    mealdb: dict = None,
    weather: dict = None,
) -> int:
    """Flattens whatever enrichment results are available into short text
    chunks and appends them to the knowledge base on disk. Skips a source
    entirely if that lookup returned nothing (None). Returns how many new
    documents were added.
    """
    if not name:
        return 0

    docs = load_kb()
    new_docs = []

    if wiki and wiki.get("extract"):
        new_docs.append(_make_doc(name, category, "wikipedia", f"{name}: {wiki['extract']}"))

    if usda and usda.get("nutrients"):
        nutrients_str = ", ".join(f"{k}: {v}" for k, v in usda["nutrients"].items())
        new_docs.append(_make_doc(
            name, category, "usda",
            f"{usda.get('description', name)} nutrition facts (USDA FoodData Central) — {nutrients_str}."
        ))

    if gbif and gbif.get("scientificName"):
        taxonomy_bits = [f"{k}: {v}" for k, v in
                          [("Scientific name", gbif.get("scientificName")), ("Kingdom", gbif.get("kingdom")),
                           ("Phylum", gbif.get("phylum")), ("Family", gbif.get("family")),
                           ("Rank", gbif.get("rank")), ("Status", gbif.get("status"))] if v]
        new_docs.append(_make_doc(name, category, "gbif", f"{name} taxonomy (GBIF) — " + "; ".join(taxonomy_bits) + "."))

    if mealdb and mealdb.get("name"):
        ingredients_str = ", ".join(mealdb.get("ingredients", [])[:12])
        new_docs.append(_make_doc(
            name, category, "themealdb",
            f"{mealdb['name']} recipe (TheMealDB, {mealdb.get('area', 'unknown cuisine')}) — "
            f"ingredients: {ingredients_str}."
        ))

    if weather and weather.get("location"):
        new_docs.append(_make_doc(
            name, category, "openweathermap",
            f"Current weather in {weather['location']}: {weather.get('description', '')}, "
            f"{weather.get('temp_c', '?')}\u00b0C."
        ))

    if not new_docs:
        return 0

    docs.extend(new_docs)
    save_kb(docs)
    return len(new_docs)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = 4, category: str = None) -> list:
    """Returns up to k relevant documents (dicts) for the query, ranked by
    TF-IDF cosine similarity, filtering out weak matches. If `category` is
    given, same-category documents are preferred but not required — a
    strong cross-category match can still surface (e.g. a place mentioned
    inside a food's description).
    """
    docs = load_kb()
    if not docs or not query or not query.strip():
        return []

    corpus = [d["text"] for d in docs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except Exception:
        logger.warning(f"TF-IDF retrieval failed for query '{query}'", exc_info=True)
        return []

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    results = []
    for doc, score in ranked:
        if score < RELEVANCE_THRESHOLD:
            continue
        results.append({**doc, "score": round(float(score), 4)})
        if len(results) >= k:
            break
    return results


def build_grounded_context(query: str, k: int = 4, category: str = None) -> str:
    """Retrieves relevant facts and formats them as a citation-style block
    ready to drop into a prompt. Returns "" if nothing relevant was found,
    so callers can skip the grounding instruction entirely in that case.
    """
    hits = retrieve(query, k=k, category=category)
    if not hits:
        return ""
    lines = [f"- ({h['source']}) {h['text']}" for h in hits]
    return "\n".join(lines)


def kb_stats() -> dict:
    docs = load_kb()
    by_source = {}
    for d in docs:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    return {"total": len(docs), "by_source": by_source}