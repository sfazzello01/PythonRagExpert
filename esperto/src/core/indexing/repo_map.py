"""Compact, deterministic repository overview injected into every prompt."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from config import REPO_MAP_FILE, REPO_MAP_MAX_CHARS

from .discovery import relative_path

logger = logging.getLogger("EspertoRepoMap")


def _is_test(filename: str) -> bool:
    return any(part.startswith("test") for part in Path(filename).parts)


def _plain_text(markup: str) -> str:
    """Drop HTML tags, images, link targets, and lines without prose (badges, link rows)."""
    text = re.sub(r"<[^>]+>", "", markup)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if len(line.split()) >= 4)


def _readme_intro(files: list[Path], max_chars: int) -> str:
    """Opening of the README closest to the repository root, cut at a line boundary."""
    readmes = [path for path in files if path.stem.lower() == "readme"]
    if not readmes:
        return ""
    readme = min(readmes, key=lambda path: (len(path.parts), path))
    text = _plain_text(readme.read_text(encoding="utf-8", errors="ignore"))
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return f"README ({relative_path(readme)}):\n{text}\n\n"


def build_repo_map(documents: list[Document], files: list[Path]) -> str:
    """List indexed files with their top-level symbols, after the README opening."""
    symbols: dict[str, list[str]] = {}
    for document in documents:
        names = symbols.setdefault(document.metadata["filename"], [])
        if document.metadata["chunk_type"] in ("function", "class"):
            names.append(document.metadata["symbol_name"])
    sources = sorted(name for name in symbols if not _is_test(name))
    test_dirs = Counter(Path(name).parent.as_posix() for name in symbols if _is_test(name))

    readme = _readme_intro(files, REPO_MAP_MAX_CHARS // 4)
    lines = {name: name for name in sources}
    # I test contano poco per orientarsi: una riga per cartella.
    lines.update({folder: f"{folder}/ ({count} file di test)" for folder, count in sorted(test_dirs.items())})
    remaining = REPO_MAP_MAX_CHARS - len(readme) - sum(len(line) + 1 for line in lines.values())

    # Lo spazio rimasto va ai simboli top-level: quanti piu' possibile per file, a tutti i file.
    def detail(name: str, max_symbols: int) -> str:
        return ": " + ", ".join(symbols[name][:max_symbols]) if symbols[name] else ""

    for max_symbols in (8, 4, 2):
        if sum(len(detail(name, max_symbols)) for name in sources) <= remaining:
            break
    for name in sources:
        if len(detail(name, max_symbols)) <= remaining:
            lines[name] += detail(name, max_symbols)
            remaining -= len(detail(name, max_symbols))

    listing = list(lines.values())
    while remaining < 0 and listing:  # repository troppo grande: l'elenco viene troncato
        remaining += len(listing.pop()) + 1
    if len(listing) < len(lines):
        listing.append(f"... (+{len(lines) - len(listing)} altri)")
    return readme + "FILE E SIMBOLI PRINCIPALI:\n" + "\n".join(listing)


def save_repo_map(documents: list[Document], files: list[Path]) -> None:
    """Persist the repository map next to the symbol graph."""
    repo_map = build_repo_map(documents, files)
    REPO_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = REPO_MAP_FILE.with_suffix(".tmp")
    temporary_file.write_text(repo_map, encoding="utf-8")
    temporary_file.replace(REPO_MAP_FILE)
    logger.info("Repository map saved: %d characters.", len(repo_map))
