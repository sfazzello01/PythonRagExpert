"""Source discovery for the codebase indexing pipeline."""

from __future__ import annotations

from pathlib import Path

from config import BASE_DIR, SKIP_DIRS, SUPPORTED_EXTENSIONS


def relative_path(path: Path) -> str:
    """Return the POSIX path relative to the indexed codebase root."""
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def service_for_path(path: Path) -> str:
    """Return the first project-directory segment for a source file."""
    try:
        return path.relative_to(BASE_DIR).parts[0]
    except (ValueError, IndexError):
        return "root"


def iter_source_files() -> list[Path]:
    """Return deterministic, supported source files under the codebase root."""
    return sorted(
        path
        for path in BASE_DIR.rglob("*")
        if path.is_file()
        and path.suffix in SUPPORTED_EXTENSIONS
        and not any(part in SKIP_DIRS for part in path.parts)
    )
