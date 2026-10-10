"""Client minimale per l'API chat di Ollama."""

from __future__ import annotations

import httpx

from config import LLM_MODEL, OLLAMA_URL, REQUEST_TIMEOUT


def chat(prompt: str, model: str = LLM_MODEL) -> str:
    """Interroga il modello con una chiamata deterministica (temperature 0, senza thinking).

    Args:
        prompt: Messaggio dell'utente.
        model: Nome del modello Ollama.

    Returns:
        Il testo della risposta.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "seed": 0},
    }
    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"]
