"""
rag_engine.py — Retrieval-Augmented Generation over your SnapKnow collection.

This is a genuine RAG pipeline, not a keyword search dressed up:
  1. Every analyzed photo's text (caption, category, name, description, details,
     fun fact) is embedded into a vector and stored in a local Chroma vector
     database.
  2. When you ask a question, the question itself is embedded the same way, and
     Chroma returns the entries whose vectors are closest to it — i.e. the
     entries most semantically relevant to what you asked, not just ones that
     share a keyword.
  3. Only those retrieved entries are handed to the LLM as context, and it's
     explicitly told to answer from that context alone.

Embeddings and generation both run through Ollama (https://ollama.com), so this
whole pipeline runs locally — no cloud vector database, no per-query API cost.

Requires Ollama running locally with two models pulled:
    ollama pull nomic-embed-text   (for embeddings)
    ollama pull llama3.2           (for answering — any Ollama chat model works)
"""

import os
import logging

logger = logging.getLogger("snapknow.rag")

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "snapknow_collection"


def _entry_to_text(result: dict) -> str:
    """Flatten one analyzed result into a single text blob for embedding."""
    return (
        f"Caption: {result.get('CAPTION', '')}\n"
        f"Category: {result.get('CATEGORY', '')}\n"
        f"Name: {result.get('NAME', '')}\n"
        f"Description: {result.get('DESCRIPTION', '')}\n"
        f"Details: {result.get('DETAILS', '')}\n"
        f"Fun fact: {result.get('FUN_FACT', '')}"
    )


def build_vectorstore(history: list, base_url: str = "http://localhost:11434",
                       embed_model: str = "nomic-embed-text"):
    """(Re)build the local vector store from every entry currently in history.
    Returns the vectorstore object, or raises if Ollama isn't reachable —
    callers should catch and show a friendly message."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    embeddings = OllamaEmbeddings(base_url=base_url, model=embed_model)

    docs = [
        Document(
            page_content=_entry_to_text(entry["result"]),
            metadata={"index": i, "caption": entry["result"].get("CAPTION", "")},
        )
        for i, entry in enumerate(history)
    ]

    # Fresh collection each build — simplest way to keep it in sync with history
    # (rebuilding a few dozen short text entries is fast; no incremental-update
    # complexity needed for a personal collection of this size).
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    try:
        existing_ids = vectordb.get()["ids"]
        if existing_ids:
            vectordb.delete(ids=existing_ids)
    except Exception:
        logger.debug("No existing Chroma collection to clear (first build)", exc_info=True)

    if docs:
        vectordb.add_documents(docs)

    return vectordb


def rag_query(question: str, vectordb, base_url: str = "http://localhost:11434",
              chat_model: str = "llama3.2", k: int = 4):
    """Retrieve the k most relevant past entries for `question`, then ask the
    LLM to answer using only that retrieved context. Returns (answer, sources)."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "Nothing in your collection looks relevant to that question yet.", []

    context = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)
    prompt = (
        "You are answering a question about a personal photo collection, using only "
        "the retrieved entries below as context. If the context doesn't actually "
        "answer the question, say so honestly rather than guessing.\n\n"
        f"RETRIEVED ENTRIES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (based only on the entries above):"
    )

    llm = ChatOllama(base_url=base_url, model=chat_model, temperature=0.2)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, retrieved_docs