"""Utility helpers for LLM-based judging via Ollama."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import THINK
from ..database.db_manipolation import DatabaseManipulator
from ..log import get_logger

# Endpoint del servizio Ollama nel network di Docker Compose.
OLLAMA_BASE_URL = "http://ollama:11434"


logger = get_logger(__name__)
Databasemanipulator = DatabaseManipulator()


def ask_ollama(prompt: str, model: str, think: bool = THINK) -> str:
    """
    Invia un prompt a Ollama e restituisce la risposta testuale del modello.

    Args:
        prompt: Testo da inviare al modello.
        model: Nome del modello Ollama da usare.
        think: Se True, il modello pensa prima di rispondere.

    Returns:
        Risposta testuale prodotta dal modello.

    Raises:
        ValueError: Se il prompt e vuoto o la risposta non e valida.
        RuntimeError: Se la chiamata HTTP a Ollama fallisce.
    """
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    base_url = OLLAMA_BASE_URL
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"Ollama HTTP error: {error.code} {error.reason}") from error
    except URLError as error:
        raise RuntimeError(f"Ollama connection error: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Invalid JSON response from Ollama") from error

    message = data.get("message", {})
    content = message.get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Empty response from Ollama")

    return content.strip()


async def stream_ollama(prompt: str, model: str) -> AsyncIterator[str]:
    """
    Invia un prompt a Ollama e restituisce i chunk testuali in streaming.

    Args:
        prompt: Testo da inviare al modello.
        model: Nome del modello Ollama da usare.

    yields:
        Porzioni incrementali di testo prodotte dal modello.
    """
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    base_url = OLLAMA_BASE_URL
    url = f"{base_url.rstrip('/')}/api/chat"  # l'url API di Ollama per le chat
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError("Invalid JSON response from Ollama") from error

                message = data.get("message", {})
                content = message.get("content", "")
                if isinstance(content, str) and content:
                    yield content  # il "return" della funzione, è un iteratore asincrono

                if data.get("done"):
                    break
    except HTTPError as error:
        raise RuntimeError(f"Ollama HTTP error: {error.code} {error.reason}") from error
    except URLError as error:
        raise RuntimeError(f"Ollama connection error: {error.reason}") from error


def _extract_json_response(raw_response: str) -> dict[str, Any]:
    """
    Estrae un oggetto JSON da una risposta del modello, anche se racchiusa in fence Markdown.

    Args:
        raw_response: Testo grezzo restituito dal modello, che dovrebbe contenere un JSON

    Returns:
        Dizionario Python rappresentante l'oggetto JSON estratto.
    """

    candidate = raw_response.strip()

    if candidate.startswith("```"):
        fence_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            candidate = fence_match.group(1).strip()

    start_index = candidate.find("{")
    end_index = candidate.rfind("}")
    if 0 <= start_index < end_index:
        candidate = candidate[start_index : end_index + 1]

    parsed_response = json.loads(candidate)
    if not isinstance(parsed_response, dict):
        raise json.JSONDecodeError("JSON response is not an object", candidate, 0)

    return parsed_response


def build_judge_prompt(parsed_text: str, gold_text: str) -> str:
    """
    Costruisce il prompt usato per il giudizio del parser.

    Args:
        parsed_text: Testo estratto dal parser.
        gold_text: Testo di riferimento (ground truth).

    Returns:
        Prompt completo da inviare al modello per ottenere un giudizio strutturato.
    """

    return (
        """
            Sei un sistema esperto di controllo qualità per il Text Extraction (OCR/Parser). Il tuo compito è valutare l'accuratezza del testo estratto (`parsed_text`) confrontandolo con il testo di riferimento (`gold_text`), che rappresenta la verità assoluta (ground truth).

            Criteri di Valutazione:
            1. **Accuratezza:** Presenza di refusi, caratteri errati, o parole allucinate/inventate.
            2. **Completezza:** Omissione di parti di testo, righe o parole presenti nel gold_text.
            3. **Fedeltà:** Alterazioni del significato originale o inversioni nell'ordine delle parole.

            Scala dei Punteggi (judge_score):
            - **5 (Eccellente):** Il parsed_text è identico al gold_text o presenta solo minime discrepanze irrilevanti (es. spazi extra).
            - **4 (Buono):** Qualche piccolo errore di trascrizione o omissione minore che non compromette il senso.
            - **3 (Sufficiente):** Errori visibili (parole troncate, brevi frasi mancanti), ma il significato generale è preservato.
            - **2 (Insufficiente):** Gravi lacune, molte parole errate o intere sezioni mancanti.
            - **1 (Pessimo):** Testo completamente distorto, illeggibile o quasi totalmente mancante/diverso.

            Vincoli di Output:
            Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza includere i tag di formattazione markdown (NO ```json) e senza alcun testo prima o dopo il JSON.

            Schema richiesto:
            {
            "judge_score": <int_tra_1_e_5>,
            "judge_feedback": "<stringa_in_italiano_di_qualche_frase>"
            }

            Nota sul feedback: deve essere un'analisi tecnica, concreta e precisa in lingua italiana che giustifichi numericamente il punteggio, evidenziando le discrepanze esatte riscontrate.
            """
        f"Dati di Input: parsed_text: {parsed_text} gold_text: {gold_text}"
    )


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    """
    Normalizza la risposta del giudice in un dizionario strutturato.

    Args:
        raw_response: Testo grezzo restituito dal modello, che dovrebbe contenere un JSON.
    Returns:
        Dizionario con chiavi 'judge_score' (int) e 'judge_feedback' (str).
    """

    try:
        parsed_response = _extract_json_response(raw_response)
    except json.JSONDecodeError:
        return {
            "judge_score": -1,
            "judge_feedback": "errore di formato JSON: " + raw_response,
        }

    score = parsed_response.get("judge_score", -3)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = -4

    feedback = parsed_response.get("judge_feedback", "")
    if not isinstance(feedback, str):
        feedback = str(feedback)

    return {"judge_score": score, "judge_feedback": feedback.strip()}


async def judge_texts(
    parsed_text: str, gold_text: str, model: str, url: str = "N/A"
) -> dict[str, Any]:
    """
    Chiede a Ollama un giudizio strutturato tra testo parsato e gold text.

    Args:
        parsed_text: Testo estratto dal parser.
        gold_text: Testo di riferimento.
        model: Nome del modello Ollama da usare.

    Returns:
        Dizionario (Json) con il giudizio del modello.
    """
    prompt = build_judge_prompt(parsed_text, gold_text)
    raw_chunks: list[str] = []
    async for chunk in stream_ollama(prompt, model):
        print(chunk, end="", flush=True)
        raw_chunks.append(chunk)

    raw_response = "".join(raw_chunks)
    judge_result = parse_judge_response(raw_response)
    judge_result["model_name"] = model

    # Salva il risultato del giudizio nel database
    

    return judge_result
