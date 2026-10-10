"""LangChain composition and streaming entry point for the Esperto RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_CONTEXT_TOKENS, LLM_MODEL, OLLAMA_URL, REPO_MAP_FILE, TEMPERATURE

from .prompts import build_prompt
from .planner import plan_subquestions
from .retrieval import Retriever
from ..shared.options import RetrievalOptions
from ..shared.text import without_imports

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Lazy singleton: evita side-effect (client Qdrant) a import-time."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _create_response_llm(use_reasoning: bool) -> ChatOllama:
    """Create a response model configured for the current user request."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        reasoning=use_reasoning,
        num_ctx=LLM_CONTEXT_TOKENS,
    )


_planner_llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.0,  # output deterministico per il planner
    reasoning=False,
    num_ctx=LLM_CONTEXT_TOKENS,  # stessa finestra della risposta: Ollama non ricarica il modello
)


def _repo_map() -> str:
    """Repository overview written by the last indexing run (empty if not available)."""
    try:
        return REPO_MAP_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


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


def _build_context(sources: list[dict]) -> str:
    """Join the sources grouped by file: files by their best-ranked chunk, chunks in source order.

    Gli import del modulo, ripetuti in ogni chunk, restano solo sul primo chunk di ogni file.
    """
    file_rank: dict[str, int] = {}
    for rank, source in enumerate(sources):
        file_rank.setdefault(source["filename"], rank)
    ordered = sorted(sources, key=lambda source: (file_rank[source["filename"]], source["start_line"] or 0))
    shown: set[str] = set()
    parts = []
    for source in ordered:
        parts.append(without_imports(source["content"]) if source["filename"] in shown else source["content"])
        shown.add(source["filename"])
    return "\n\n---\n\n".join(parts)


async def retrieve_only(
    question: str,
    use_planner: bool = True,
    options: RetrievalOptions = RetrievalOptions(),
    subquestions: list[str] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Run planner + retrieval without generation; returns (subquestions, sources, dropped).

    Con `subquestions` il planner non viene chiamato: si riusa un piano gia' calcolato.
    `dropped` sono i candidati rimasti fuori dal contesto (diagnostica per il benchmark).
    """
    if subquestions is None:
        subquestions = (
            await plan_subquestions(question, _planner_llm, _repo_map() if options.use_repo_map else "")
            if use_planner
            else [question]
        )
    sources, dropped = await _get_retriever().retrieve(subquestions, question, options)
    return subquestions, sources, dropped


async def stream_expert_response(
    question: str,
    use_planner: bool = True,
    use_reasoning: bool = False,
    options: RetrievalOptions = RetrievalOptions(),
):
    """Yield sources metadata then generated answer chunks for the API streaming response."""
    subquestions, sources, _ = await retrieve_only(question, use_planner, options)
    context = _build_context(sources)
    repo_map = _repo_map() if options.use_repo_map else ""
    yield json.dumps(
        {"type": "sources", "data": {"sources": sources, "subquestions": subquestions, "repo_map": repo_map}},
        ensure_ascii=False,
    ) + "\n"
    repo_map_section = (
        "MAPPA DEL REPOSITORY (per orientarsi e per le domande generali; "
        f"i dettagli vanno presi dal codice recuperato):\n{repo_map}\n\n"
        if repo_map
        else ""
    )
    prompt_value = build_prompt().invoke(
        {"repo_map": repo_map_section, "context": context, "question": question}
    )
    _save_last_prompt(prompt_value)
    async for chunk in _create_response_llm(use_reasoning).astream(prompt_value):
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            yield json.dumps({"type": "reasoning", "data": reasoning}, ensure_ascii=False) + "\n"
        if chunk.content:
            yield json.dumps({"type": "answer", "data": chunk.content}, ensure_ascii=False) + "\n"