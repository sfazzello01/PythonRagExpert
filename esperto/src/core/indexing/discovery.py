"""Source discovery for the codebase indexing pipeline."""

from __future__ import annotations

from pathlib import Path

from ..shared.settings import BASE_DIR, INDEX_TARGET_PATHS, SKIP_DIRS, SUPPORTED_EXTENSIONS


def service_for_path(path: Path) -> str:
    """Return the first project-directory segment for a source file."""
    try:
        relative_path = path.relative_to(BASE_DIR)
    except ValueError:
        return "root"
    return relative_path.parts[0] if relative_path.parts else "root"


def index_targets() -> tuple[Path, ...]:
    """Resolve the configured source targets relative to the project root."""
    return tuple(BASE_DIR / target for target in INDEX_TARGET_PATHS)


def iter_source_files() -> list[Path]:
    """Return deterministic, supported source files from configured index targets."""
    files: list[Path] = []
    for target in index_targets():
        if target.is_file():
            if target.suffix in SUPPORTED_EXTENSIONS:
                files.append(target)
            continue
        if not target.is_dir():
            continue
        for path in target.rglob("*"):
            if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)