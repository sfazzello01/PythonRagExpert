"""Scoperta dei file sorgente da indicizzare."""

from __future__ import annotations

from pathlib import Path

from config import BASE_DIR, SKIP_DIRS, SUPPORTED_EXTENSIONS


def relative_path(path: Path) -> str:
    """Calcola il percorso POSIX di un file rispetto alla radice della codebase indicizzata.

    Args:
        path: Percorso del file.

    Returns:
        Il percorso relativo a BASE_DIR, oppure il percorso invariato se il file ne è fuori.
    """
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def service_for_path(path: Path) -> str:
    """Ricava il servizio a cui appartiene un file: il primo segmento del suo percorso relativo.

    Args:
        path: Percorso del file.

    Returns:
        Il primo segmento del percorso relativo a BASE_DIR, oppure `root` se il file ne è fuori.
    """
    try:
        return path.relative_to(BASE_DIR).parts[0]
    except (ValueError, IndexError):
        return "root"


def iter_source_files() -> list[Path]:
    """Elenca i file sorgente supportati sotto la radice della codebase.

    Returns:
        I percorsi in ordine alfabetico, senza le cartelle escluse e le estensioni non supportate.
    """
    return sorted(
        path
        for path in BASE_DIR.rglob("*")
        if path.is_file()
        and path.suffix in SUPPORTED_EXTENSIONS
        and not any(part in SKIP_DIRS for part in path.parts)
    )
