"""Bounded LLM planning for multi-part codebase questions."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..shared.text import sanitize_text

logger = logging.getLogger("EspertoPlanner")

_PLANNER_PROMPT = """Sei un planner per il retrieval di una codebase.
Scomponi la domanda in una o due sotto-domande tecniche, indipendenti e mirate al codice.
Mantieni una sola domanda se e gia specifica. Non rispondere alla domanda.
Restituisci esclusivamente JSON valido: {{"subquestions":["..."]}}.

DOMANDA: {question}"""


def parse_subquestions(content: str, original_question: str) -> list[str]:
    """Validate planner output and fall back to the original question."""
    cleaned_original = sanitize_text(original_question).strip()
    try:
        json_start = content.index("{")
        payload, _ = json.JSONDecoder().raw_decode(content[json_start:])
        candidates = payload.get("subquestions", [])
    except (ValueError, json.JSONDecodeError, AttributeError):
        logger.warning(
            "Planner returned invalid JSON; using original question. Output: %r",
            content[:300],
        )
        return [cleaned_original] if cleaned_original else []
    if not isinstance(candidates, list):
        return [cleaned_original] if cleaned_original else []

    questions: list[str] = []
    seen: set[str] = set()
    for candidate in candidates[:2]:
        if not isinstance(candidate, str):
            continue
        question = sanitize_text(candidate).strip()
        normalized = question.casefold()
        if len(question) >= 3 and normalized not in seen:
            questions.append(question)
            seen.add(normalized)
    return questions or ([cleaned_original] if cleaned_original else [])


async def plan_subquestions(question: str, llm: Any) -> list[str]:
    """Ask the local LLM for a small retrieval plan without blocking generation."""
    try:
        response = await llm.ainvoke(_PLANNER_PROMPT.format(question=question))
        content = response.content if isinstance(response.content, str) else ""
    except Exception as error:
        logger.warning("Planner failed; using original question: %s", error)
        return [question]
    subquestions = parse_subquestions(content, question)
    logger.info("Planner produced %d retrieval question(s).", len(subquestions))
    return subquestions