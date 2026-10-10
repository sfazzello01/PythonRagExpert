"""Funzioni di utilità sul testo condivise da indicizzazione e retrieval."""

import re

_IMPORTS_BLOCK = re.compile(r"--- IMPORTS DI CONTESTO ---\n.*?(?=--- CODICE SORGENTE ---\n)", re.DOTALL)


def sanitize_text(text: str) -> str:
    """Rimuove dal testo le sequenze non codificabili in UTF-8, prima di passarlo ai modelli.

    Args:
        text: Testo da ripulire.

    Returns:
        Il testo senza i caratteri non validi.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


def without_imports(chunk: str) -> str:
    """Toglie da un chunk di codice il blocco degli import del modulo.

    Il contesto lo mostra una sola volta per file.

    Args:
        chunk: Testo del chunk.

    Returns:
        Il chunk senza il blocco degli import.
    """
    return _IMPORTS_BLOCK.sub("", chunk, count=1)
