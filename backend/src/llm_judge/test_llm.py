"""Script interattivo per testare il giudice LLM via Ollama."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import AsyncGenerator


from config import (
    GEMMA,
    LLAMA,
    MINISTRAL,
    PHI,
    QWEN,
)

GEMMA, LLAMA, PHI, MINISTRAL, QWEN  # cosi non da warning di variabile non usata


try:
    from .judge import (
        build_judge_prompt,
        parse_judge_response,
        stream_ollama,
    )
except ImportError:
    from judge import (
        build_judge_prompt,
        parse_judge_response,
        stream_ollama,
    )


DEFAULT_PARSED_PATH = Path(__file__).with_name("parsed.txt")
DEFAULT_GS_PATH = Path(__file__).with_name("gs_test.json")


def load_text_file(path: Path) -> str:
    """Carica testo grezzo da un file .txt."""
    with path.open(encoding="utf-8") as file_handle:
        return file_handle.read().strip()


def load_gold_text(path: Path) -> str:
    """Carica gold_text dal file JSON indicato."""
    with path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, dict):
        raise ValueError("il file JSON deve contenere un oggetto")

    gold_text = data.get("gold_text", "")
    if not isinstance(gold_text, str) or not gold_text.strip():
        raise ValueError("gold_text non trovato o vuoto")

    return gold_text.strip()


async def ask_general_question(model: str) -> AsyncGenerator[str, None]:
    """Chiede una domanda generica al LLM."""
    question = input("Fai una domanda al LLM: ").strip()

    async for chunk in stream_ollama(question, model):
        yield chunk


async def test_judge_texts(
    parsed_text: str, gold_text: str, model: str
) -> AsyncGenerator[str, None]:
    """Testa la funzione judge_texts con i dati forniti."""
    prompt = build_judge_prompt(parsed_text=parsed_text, gold_text=gold_text)
    async for chunk in stream_ollama(prompt, model):
        yield chunk


def interactive_choices() -> tuple[str, str]:
        models = [GEMMA, LLAMA, PHI, MINISTRAL, QWEN]
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        model_input = input(
            "Scegli il modello da usare (1-5, default: GEMMA): "
        ).strip()
        if model_input and model_input.isdigit() and 1 <= int(model_input) <= 5:
            model = models[int(model_input) - 1]
        else:
            model = GEMMA
        print(f"Modello selezionato: {model}\n")
        inp = input(
            "premi (1) per chiedere qualcosa al llm, premi (2) per testare il giudizio al parser: "
        )
        if inp not in {"1", "2"}:
            print("Input non valido, default a 2")
            inp = "2"
        return inp, model

async def main() -> None:
    """Avvia una sessione interattiva per testare judge_texts."""
    print("LLM judge interattivo")

    parsed_path_input = input(f"File parsed TXT [{DEFAULT_PARSED_PATH.name}]: ").strip()
    parsed_path = Path(parsed_path_input) if parsed_path_input else DEFAULT_PARSED_PATH
    gs_path_input = input(f"File gold JSON [{DEFAULT_GS_PATH.name}]: ").strip()
    gs_path = Path(gs_path_input) if gs_path_input else DEFAULT_GS_PATH

    try:
        parsed_text = load_text_file(parsed_path)
        if not parsed_text:
            raise ValueError("parsed_text vuoto")
        gold_text = load_gold_text(gs_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Impossibile leggere i file di input: {error}")
        return

    while 1:

        inp, model = interactive_choices()
        if inp == "1":
            prompt = input("scrivi prompt: ")
            print("generando risposta...")
            start_time = time.time()

            async for chunk in stream_ollama(prompt, model):
                print(chunk, end="", flush=True)
            print()

            end_time = time.time()
            tempo = end_time - start_time
            print("tempo totale: {:.2f} secondi".format(tempo))

        if inp == "2":
            print("giudizio in corso...")
            start_time = time.time()

            risposta_chunks: list[str] = []
            async for chunk in test_judge_texts(parsed_text, gold_text, model):
                risposta_chunks.append(chunk)
                print(chunk, end="", flush=True)
            print()

            raw_response = "".join(risposta_chunks)
            print(parse_judge_response(raw_response))
            print()

            end_time = time.time()
            tempo = end_time - start_time
            print("tempo totale: {:.2f} secondi".format(tempo))


if __name__ == "__main__":
    asyncio.run(main())
