"""LLM-as-judge a verdetti binari ancorati a fatti di riferimento."""

from __future__ import annotations

import json
from typing import Callable, Optional

from config import JUDGE_CONTEXT_MAX_CHARS

from . import ollama

_SYSTEM = (
    "You are a strict grader of answers about a Python codebase. "
    "Answer only with the requested JSON."
)

_FACTS_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["verdicts"],
}
_BOOL_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "boolean"}},
    "required": ["value"],
}

Chat = Callable[..., str]


def _parse(raw: str) -> Optional[dict]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judge_facts(question: str, answer: str, key_facts: list[str], chat: Chat = ollama.chat) -> Optional[list[bool]]:
    """Per ogni fatto: la risposta lo afferma (o lo implica chiaramente)? None se il giudice fallisce."""
    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(key_facts, start=1))
    prompt = (
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nFACTS:\n{numbered}\n\n"
        "For each fact, in order, output true if the ANSWER states it or clearly implies it, "
        "false otherwise. Do not use outside knowledge. "
        'Return {"verdicts": [...]} with exactly one boolean per fact.'
    )
    parsed = _parse(chat(prompt, system=_SYSTEM, schema=_FACTS_SCHEMA))
    verdicts = parsed.get("verdicts") if parsed else None
    if not isinstance(verdicts, list) or len(verdicts) != len(key_facts):
        return None
    return [bool(v) for v in verdicts]


def judge_faithful(question: str, answer: str, context: str, chat: Chat = ollama.chat) -> Optional[bool]:
    """True se ogni affermazione sul codice nella risposta e' supportata dal contesto."""
    prompt = (
        f"CONTEXT (code excerpts):\n{context[:JUDGE_CONTEXT_MAX_CHARS]}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Is every claim the ANSWER makes about the code supported by the CONTEXT? "
        "An answer that only says the information is not available is supported. "
        'Return {"value": true} or {"value": false}.'
    )
    parsed = _parse(chat(prompt, system=_SYSTEM, schema=_BOOL_SCHEMA))
    return bool(parsed["value"]) if parsed and "value" in parsed else None


def judge_abstains(question: str, answer: str, chat: Chat = ollama.chat) -> Optional[bool]:
    """True se la risposta dichiara che l'informazione non e' disponibile nel codice."""
    prompt = (
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Does the ANSWER state that the requested information is not available, not visible, "
        "or cannot be determined from the provided code (instead of answering it)? "
        'Return {"value": true} or {"value": false}.'
    )
    parsed = _parse(chat(prompt, system=_SYSTEM, schema=_BOOL_SCHEMA))
    return bool(parsed["value"]) if parsed and "value" in parsed else None
