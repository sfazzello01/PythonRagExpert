"""Client minimale per l'API chat di Ollama."""

from __future__ import annotations

from typing import Optional

import httpx

from config import LLM_MODEL, OLLAMA_URL, REQUEST_TIMEOUT


def chat(
    prompt: str, system: Optional[str] = None, schema: Optional[dict] = None, model: str = LLM_MODEL
) -> str:
    """Una chiamata deterministica (temperature 0, senza thinking); `schema` vincola l'output JSON."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "seed": 0},
    }
    if schema:
        payload["format"] = schema
    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"]
