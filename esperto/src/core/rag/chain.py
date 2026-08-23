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
from .planner import plan_subquestions
from .retrieval import HybridRetriever

_retriever = HybridRetriever()
def _create_response_llm(use_reasoning: bool) -> ChatOllama:
    """Create a response model configured for the current user request."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        reasoning=use_reasoning,
    )


_llm = _create_response_llm(False)
_planner_llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=TEMPERATURE,
    format="json",
    reasoning=False,
)
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


async def stream_expert_response(
    question: str,
    use_planner: bool = True,
    include_codebase_map: bool = False,
    use_reasoning: bool = False,
):
    """Yield sources metadata then generated answer chunks for the API streaming response."""
    subquestions = (
        await plan_subquestions(question, _planner_llm)
        if use_planner
        else [question]
    )
    context, sources = await _retriever.retrieve_many_with_sources(
        subquestions,
        question,
        include_codebase_map=include_codebase_map,
    )
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