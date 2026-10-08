"""Compact, deterministic repository overview injected into every prompt."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

from config import REPO_MAP_FILE, REPO_MAP_MAX_CHARS

from .discovery import relative_path

logger = logging.getLogger("EspertoRepoMap")

# Sottoclassi elencate al massimo per ogni classe base.
MAX_SUBCLASSES = 10
_HIERARCHY_TITLE = "GERARCHIA DELLE CLASSI (classe base <- sottoclassi dirette):\n"


def _is_test(filename: str) -> bool:
    return any(part.startswith("test") for part in Path(filename).parts)


def _readme_intro(files: list[Path], max_chars: int) -> str:
    """Opening of the README closest to the repository root, after its leading badges and HTML."""
    readmes = [path for path in files if path.stem.lower() == "readme"]
    if not readmes:
        return ""
    readme = min(readmes, key=lambda path: (len(path.parts), path))
    lines = readme.read_text(encoding="utf-8", errors="ignore").splitlines()
    while lines and lines[0].lstrip()[:1] in ("", "<", "[", "!"):  # righe vuote, HTML, badge
        lines.pop(0)
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return f"README ({relative_path(readme)}):\n{text}\n\n"


def _folders(files: list[Path], max_chars: int) -> str:
    """One line per folder with its file count, at the deepest level that fits the budget."""
    parents = [Path(relative_path(path)).parent.parts for path in files]
    for depth in range(max(map(len, parents), default=0), -1, -1):
        counts = Counter("/".join(parts[:depth]) or "." for parts in parents)
        listing = "\n".join(f"{folder}/ ({count} file)" for folder, count in sorted(counts.items()))
        if len(listing) <= max_chars:
            break
    return f"CARTELLE:\n{listing}\n\n"


def _class_hierarchy(graph: dict, max_chars: int) -> str:
    """Base classes with their direct subclasses, the most extended first; tests are left out."""
    classes = {
        node["id"]: node["name"]
        for node in graph["nodes"]
        if node["kind"] == "class" and not _is_test(node["filename"])
    }
    subclasses: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge["kind"] == "inherits" and edge["source"] in classes and edge["target"] in classes:
            subclasses[edge["target"]].append(classes[edge["source"]])
    lines = []
    for base in sorted(subclasses, key=lambda base: (-len(subclasses[base]), classes[base])):
        names = sorted(set(subclasses[base]))
        extra = f", ... (+{len(names) - MAX_SUBCLASSES})" if len(names) > MAX_SUBCLASSES else ""
        lines.append(f"{classes[base]} <- {', '.join(names[:MAX_SUBCLASSES])}{extra}")
    while lines and len(_HIERARCHY_TITLE) + sum(len(line) + 1 for line in lines) > max_chars:
        lines.pop()
    return _HIERARCHY_TITLE + "\n".join(lines) if lines else ""


def build_repo_map(files: list[Path], graph: dict) -> str:
    """README opening, folders, and class hierarchy, within REPO_MAP_MAX_CHARS."""
    text = _readme_intro(files, REPO_MAP_MAX_CHARS // 4)
    text += _folders(files, REPO_MAP_MAX_CHARS // 2 - len(text))
    return (text + _class_hierarchy(graph, REPO_MAP_MAX_CHARS - len(text))).rstrip()


def save_repo_map(files: list[Path], graph: dict) -> None:
    """Persist the repository map next to the symbol graph."""
    repo_map = build_repo_map(files, graph)
    REPO_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = REPO_MAP_FILE.with_suffix(".tmp")
    temporary_file.write_text(repo_map, encoding="utf-8")
    temporary_file.replace(REPO_MAP_FILE)
    logger.info("Repository map saved: %d characters.", len(repo_map))
