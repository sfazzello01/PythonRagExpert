"""LangChain composition and streaming entry point for the Esperto RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_MODEL, OLLAMA_URL, TEMPERATURE

from .prompts import build_prompt
from .planner import plan_subquestions
from .retrieval import HybridRetriever

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Lazy singleton: evita side-effect (client Qdrant) a import-time."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _create_response_llm(use_reasoning: bool) -> ChatOllama:
    """Create a response model configured for the current user request."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        reasoning=use_reasoning,
    )


_planner_llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.0,  # output deterministico per il planner
    reasoning=False,
)


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


async def stream_expert_response(
    question: str,
    use_planner: bool = True,
    use_reasoning: bool = False,
):
    """Yield sources metadata then generated answer chunks for the API streaming response."""
    subquestions = (
        await plan_subquestions(question, _planner_llm)
        if use_planner
        else [question]
    )
    context, sources = await _get_retriever().retrieve_many_with_sources(subquestions, question)
    yield json.dumps(
        {"type": "sources", "data": {"sources": sources, "subquestions": subquestions}},
        ensure_ascii=False,
    ) + "\n"
    prompt_value = build_prompt().invoke({"context": context, "question": question})
    _save_last_prompt(prompt_value)
    async for chunk in _create_response_llm(use_reasoning).astream(prompt_value):
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            yield json.dumps({"type": "reasoning", "data": reasoning}, ensure_ascii=False) + "\n"
        if chunk.content:
            yield json.dumps({"type": "answer", "data": chunk.content}, ensure_ascii=False) + "\n"