"""Architecture-map document generation for cross-module questions."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

from ..shared.settings import BASE_DIR, SKIP_DIRS, SUPPORTED_EXTENSIONS
from .discovery import index_targets


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def extract_fastapi_routes(path: Path) -> list[str]:
    """Extract decorated FastAPI route names from a Python source file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []
    routes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            decorator_text = ast.unparse(decorator)
            if any(method in decorator_text for method in ("get(", "post(", "put(", "delete(", "patch(")):
                routes.append(f"{decorator_text} -> {node.name}()")
    return routes


def extract_frontend_fetches(path: Path) -> list[str]:
    """Extract frontend fetch calls with their line numbers."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    fetches: list[str] = []
    for match in re.finditer(r"fetch\s*\(\s*([^,\n)]+)", text):
        endpoint = match.group(1).strip().strip("'`\"")
        line = text.count("\n", 0, match.start()) + 1
        fetches.append(f"linea {line}: fetch({endpoint})")
    return fetches


def generate_target_tree(path: Path, prefix: str = "") -> list[str]:
    """Generate a filtered tree of indexable source files."""
    try:
        items = sorted(
            (item for item in path.iterdir() if item.name not in SKIP_DIRS),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
    except OSError:
        return []

    lines: list[str] = []
    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = "`-- " if is_last else "|-- "
        if item.is_dir():
            lines.append(f"{prefix}{connector}{item.name}/")
            lines.extend(generate_target_tree(item, prefix + ("    " if is_last else "|   ")))
        elif item.suffix in SUPPORTED_EXTENSIONS or item.name in {"docker-compose.yaml", "domains.json"}:
            lines.append(f"{prefix}{connector}{item.name}")
    return lines


def build_codebase_summary_documents(all_files: list[Path]) -> list[Document]:
    """Build the global architecture summary document used for broad questions."""
    sections: list[str] = []
    total_python_files = 0
    total_routes = 0
    total_fetches = 0
    for path in all_files:
        details: list[str] = []
        if path.suffix == ".py":
            total_python_files += 1
            routes = extract_fastapi_routes(path)
            if routes:
                details.append("ROUTE FASTAPI:\n  " + "\n  ".join(routes))
                total_routes += len(routes)
        if path.suffix in {".html", ".py", ".ts", ".js"}:
            fetches = extract_frontend_fetches(path)
            if fetches:
                details.append("CHIAMATE HTTP/FETCH:\n  " + "\n  ".join(fetches))
                total_fetches += len(fetches)
        if details:
            sections.append(f"FILE: {_relative_path(path)}\n" + "\n".join(details))

    tree_lines: list[str] = []
    for target in index_targets():
        if not target.exists():
            continue
        tree_lines.append(f"`-- {target.name}/" if target.is_dir() else f"`-- {target.name}")
        if target.is_dir():
            tree_lines.extend(generate_target_tree(target, "    "))

    overview = (
        "==================================================================\n"
        "             OVERVIEW GENERALE DELLA ARCHITETTURA RAG\n"
        "==================================================================\n\n"
        f"METRICHE CHIAVE:\n- File indicizzati: {len(all_files)}\n"
        f"- Moduli Python: {total_python_files}\n"
        f"- Endpoint API esposti: {total_routes}\n"
        f"- Chiamate Fetch Frontend: {total_fetches}\n\n"
        "ALBERO DELLE CARTELLE:\n"
        + "\n".join(tree_lines)
        + "\n\n==================================================================\n"
        "                 MAPPA ROUTE E CHIAMATE DEI FILE\n"
        "==================================================================\n\n"
        + "\n\n".join(sections)
    )
    return [
        Document(
            page_content=overview,
            metadata={
                "filename": "__codebase_map__",
                "service": "architecture",
                "chunk_type": "architecture_map",
                "chunk_id": hashlib.md5("global_codebase_map_sota".encode("utf-8")).hexdigest(),
            },
        )
    ]