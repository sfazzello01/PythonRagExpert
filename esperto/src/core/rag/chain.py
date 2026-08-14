"""LangChain composition and streaming entry point for the Esperto RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_MODEL, OLLAMA_URL, TEMPERATURE

from .prompts import build_prompt
from .retrieval import HybridRetriever

_retriever = HybridRetriever()
_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
_generation_chain = build_prompt() | _llm | StrOutputParser()


def _save_last_prompt(prompt_value) -> None:
    """Persist the rendered prompt for local inspection after a UI request."""
    prompt_path = os.getenv("PROMPT_DEBUG_FILE")
    if not prompt_path:
        return

    payload = {
        "messages": [
            {"role": message.type, "content": message.content}
            for message in prompt_value.messages
        ]
    }
    destination = Path(prompt_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(destination)


def build_rag_chain():
    """Compose retrieval, prompt construction, Ollama generation, and string streaming."""
    return (
        {"context": _retriever.retrieve_context, "question": RunnablePassthrough()}
        | build_prompt()
        | _llm
        | StrOutputParser()
    )


rag_chain = build_rag_chain()


async def stream_expert_response(question: str):
    """Yield sources metadata then generated answer chunks for the API streaming response."""
    context, sources = await _retriever.retrieve_with_sources(question)
    yield f"SOURCES:{json.dumps({'sources': sources}, ensure_ascii=False)}\n"
    prompt_value = build_prompt().invoke({"context": context, "question": question})
    _save_last_prompt(prompt_value)
    async for chunk in (_llm | StrOutputParser()).astream(prompt_value):
        yield chunk