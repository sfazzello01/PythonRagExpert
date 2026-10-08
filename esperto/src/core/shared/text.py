"""Text utilities shared by indexing and retrieval."""

import re

_IMPORTS_BLOCK = re.compile(r"--- IMPORTS DI CONTESTO ---\n.*?(?=--- CODICE SORGENTE ---\n)", re.DOTALL)


def sanitize_text(text: str) -> str:
    """Remove invalid UTF-8 sequences from text before model processing."""
    return text.encode("utf-8", "ignore").decode("utf-8")


def without_imports(chunk: str) -> str:
    """Drop the module-imports block of a code chunk: the context shows it once per file."""
    return _IMPORTS_BLOCK.sub("", chunk, count=1)
