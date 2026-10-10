"""Composizione LangChain della pipeline RAG e punto di ingresso dello streaming."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.prompt_values import PromptValue
from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_CONTEXT_TOKENS, LLM_MODEL, OLLAMA_URL, REPO_MAP_FILE, TEMPERATURE

from ..shared.options import RetrievalOptions
from ..shared.text import without_imports
from .planner import plan_subquestions
from .prompts import build_prompt
from .retrieval import Retriever

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Restituisce il retriever condiviso, creato alla prima richiesta.

    La creazione pigra evita di aprire il client Qdrant all'import del modulo.

    Returns:
        L'unica istanza di `Retriever` del processo.
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _create_response_llm(use_reasoning: bool) -> ChatOllama:
    """Crea il modello che genera la risposta, configurato per la richiesta corrente.

    Args:
        use_reasoning: True per far produrre al modello anche il ragionamento.

    Returns:
        Il client del modello Ollama.
    """
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
    """Legge la mappa del repository scritta dall'ultima indicizzazione.

    Returns:
        Il testo della mappa, oppure una stringa vuota se non è disponibile.
    """
    try:
        return REPO_MAP_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def _save_last_prompt(prompt_value: PromptValue) -> None:
    """Salva il prompt inviato al modello, per poterlo ispezionare dopo una richiesta.

    Non fa nulla se la variabile d'ambiente PROMPT_DEBUG_FILE non è impostata.

    Args:
        prompt_value: Prompt già composto.
    """
    prompt_path = os.getenv("PROMPT_DEBUG_FILE")
    if not prompt_path:
        return

    payload = {
        "messages": [
            {"role": message.type, "content": message.content}
            for message in prompt_value.to_messages()
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
    """Unisce le fonti raggruppandole per file.

    I file seguono l'ordine del loro chunk migliore e i chunk di un file l'ordine del sorgente. Gli
    import del modulo, ripetuti in ogni chunk, restano solo sul primo chunk di ogni file.

    Args:
        sources: Fonti in ordine di rilevanza.

    Returns:
        Il contesto da inserire nel prompt.
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
    """Esegue planner e retrieval senza generare la risposta.

    Args:
        question: Domanda dell'utente.
        use_planner: True per scomporre la domanda con il planner.
        options: Interruttori degli stadi di retrieval.
        subquestions: Piano già calcolato; se presente il planner non viene chiamato.

    Returns:
        Le sotto-domande usate, le fonti del contesto e i candidati rimasti fuori dal contesto
        (diagnostica per il benchmark).
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
) -> AsyncIterator[str]:
    """Risponde alla domanda trasmettendo prima le fonti e poi il testo generato.

    Args:
        question: Domanda dell'utente.
        use_planner: True per scomporre la domanda con il planner.
        use_reasoning: True per trasmettere anche il ragionamento del modello.
        options: Interruttori degli stadi di retrieval.

    Yields:
        Righe JSON con un evento ciascuna: `sources` (fonti, sotto-domande e mappa), poi `reasoning` e
        `answer` a mano a mano che il modello genera.
    """
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
