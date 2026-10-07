"""Bounded LLM planning for multi-part codebase questions."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..shared.text import sanitize_text

logger = logging.getLogger("EspertoPlanner")

_PLANNER_PROMPT = """Sei un planner per il retrieval di una codebase.
Genera sotto-domande solo se migliorano chiaramente il retrieval.
{grounding}
Ogni sotto-domanda deve essere rispondibile cercando nella codebase.
Se la domanda è già specifica restituisci solo la domanda stessa; altrimenti genera
da due a quattro sotto-domande a seconda della sua complessità. Non rispondere alla domanda.
Le sotto-domande devono essere sempre nella stessa lingua usata nella domanda (ad esempio, se la domanda è in inglese, le sotto-domande devono essere in inglese).
Restituisci esclusivamente JSON valido: {{"subquestions":["..."]}}.

{repo_map}DOMANDA: {question}"""

_BLIND_GROUNDING = "Non introdurre ipotesi, nomi di file, simboli o tecnologie non presenti nella domanda."
_MAP_GROUNDING = (
    "Usa la mappa del repository per ancorare le sotto-domande a file e simboli reali e pertinenti.\n"
    "Non introdurre ipotesi, nomi di file, simboli o tecnologie non presenti nella domanda o nella mappa."
)


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
    for candidate in candidates[:4]:
        if not isinstance(candidate, str):
            continue
        question = sanitize_text(candidate).strip()
        normalized = question.casefold()
        if len(question) >= 3 and normalized not in seen:
            questions.append(question)
            seen.add(normalized)
    return questions or ([cleaned_original] if cleaned_original else [])


async def plan_subquestions(question: str, llm: Any, repo_map: str = "") -> list[str]:
    """Ask the local LLM for a small retrieval plan, grounded on the repository map when given."""
    prompt = _PLANNER_PROMPT.format(
        grounding=_MAP_GROUNDING if repo_map else _BLIND_GROUNDING,
        repo_map=f"MAPPA DEL REPOSITORY:\n{repo_map}\n\n" if repo_map else "",
        question=question,
    )
    try:
        response = await llm.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else ""
    except Exception as error:
        logger.warning("Planner failed; using original question: %s", error)
        return [question]
    subquestions = parse_subquestions(content, question)
    logger.info("Planner produced %d retrieval question(s).", len(subquestions))
    return subquestions