from __future__ import annotations

import time
from pathlib import Path
import ast
import re

from langchain_core.documents import Document  # type: ignore
from langchain_qdrant import QdrantVectorStore  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

from config import (
    EMBEDDING_MODEL,
    INDEX_CHUNK_OVERLAP,
    INDEX_CHUNK_SIZE,
    INDEX_MAX_FILE_CHARS,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    SUPPORTED_EXTENSIONS,
)
from .nomic_conf import NomicOllamaEmbeddings, sanitize_text


BASE_DIR = Path("/app")
INDEX_TARGETS = (
    BASE_DIR / "backend",
    BASE_DIR / "frontend",
    BASE_DIR / "src",
    BASE_DIR / "config.py",
    BASE_DIR / "domains.json",
    BASE_DIR / "docker-compose.yaml",
)
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "logs",
    "mariadb_data",
    "ollama_data",
    "qdrant_data",
    "node_modules",
}


def _wait_for_qdrant() -> None:
    client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
    last_error: Exception | None = None

    for _ in range(30):
        try:
            client.get_collections()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)

    raise RuntimeError(f"Qdrant non disponibile: {last_error}")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []

    for target in INDEX_TARGETS:
        if target.is_file():
            if target.suffix in SUPPORTED_EXTENSIONS:
                files.append(target)
            continue

        if not target.is_dir():
            continue

        for path in target.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            files.append(path)

    return sorted(files)


def _service_for_path(path: Path) -> str:
    try:
        relative = path.relative_to(BASE_DIR)
    except ValueError:
        return "root"

    return relative.parts[0] if relative.parts else "root"


def _load_documents() -> list[Document]:
    documents: list[Document] = []

    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Skip {path}: {exc}", flush=True)
            continue

        text = sanitize_text(text).strip()
        if not text:
            continue

        if len(text) > INDEX_MAX_FILE_CHARS:
            text = text[:INDEX_MAX_FILE_CHARS]

        relative_path = str(path.relative_to(BASE_DIR))
        service = _service_for_path(path)
        indexed_text = (
            f"FILE: {relative_path}\n"
            f"SERVIZIO: {service}\n"
            f"ESTENSIONE: {path.suffix}\n\n"
            f"{text}"
        )

        documents.append(
            Document(
                page_content=indexed_text,
                metadata={
                    "filename": relative_path,
                    "service": service,
                },
            )
        )

    documents.extend(_build_codebase_summary_documents())
    return documents


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator

    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return f"{target.value.id}.{target.attr}"

    return None


def _extract_fastapi_routes(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []

    routes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            name = _decorator_name(decorator)
            if name not in {
                "app.get",
                "app.post",
                "app.delete",
                "app.put",
                "app.patch",
            }:
                continue

            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue

            route_arg = decorator.args[0]
            if not isinstance(route_arg, ast.Constant):
                continue

            route = str(route_arg.value)
            method = name.split(".")[-1].upper()
            routes.append(f"{method} {route} -> {node.name}()")

    return routes


def _extract_python_symbols(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                symbols.append(f"classe {node.name} metodi: {', '.join(methods)}")
            else:
                symbols.append(f"classe {node.name}")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [
                name
                for decorator in node.decorator_list
                if (name := _decorator_name(decorator))
            ]
            decorator_text = f" decoratori: {', '.join(decorators)}" if decorators else ""
            symbols.append(f"funzione {node.name}{decorator_text}")

    return symbols


def _extract_frontend_fetches(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    fetches: list[str] = []
    for match in re.finditer(r"fetch\(([^,\n)]+)", text):
        endpoint = match.group(1).strip()
        line = text.count("\n", 0, match.start()) + 1
        fetches.append(f"linea {line}: fetch({endpoint})")

    return fetches


def _build_codebase_summary_documents() -> list[Document]:
    sections: list[str] = []

    for path in _iter_source_files():
        relative_path = str(path.relative_to(BASE_DIR))
        details: list[str] = []

        if path.suffix == ".py":
            symbols = _extract_python_symbols(path)
            routes = _extract_fastapi_routes(path)

            if symbols:
                details.append("SIMBOLI PYTHON:")
                details.extend(symbols)

            if routes:
                details.append("ROUTE FASTAPI:")
                details.extend(routes)

        if path.suffix in {".html", ".py"}:
            fetches = _extract_frontend_fetches(path)
            if fetches:
                details.append("CHIAMATE HTTP/FETCH:")
                details.extend(fetches)

        if details:
            sections.append(f"FILE: {relative_path}\n" + "\n".join(details))

    if not sections:
        return []

    return [
        Document(
            page_content=(
                "MAPPA GENERALE DELLA CODEBASE\n"
                "Questo documento sintetico collega file, moduli, funzioni, "
                "classi, route FastAPI e chiamate HTTP/fetch. Serve come indice "
                "di orientamento per qualunque domanda sulla codebase.\n\n"
                + "\n\n".join(sections)
            ),
            metadata={
                "filename": "__codebase_map__",
                "service": "architecture",
            },
        )
    ]


def build_index() -> None:
    _wait_for_qdrant()

    documents = _load_documents()
    if not documents:
        raise RuntimeError("Nessun documento trovato per indicizzare la codebase")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=INDEX_CHUNK_SIZE,
        chunk_overlap=INDEX_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = NomicOllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

    last_error: Exception | None = None
    for attempt in range(1, 31):
        try:
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=QDRANT_COLLECTION,
                url=f"http://{QDRANT_HOST}:6333",
                force_recreate=True,
            )
            break
        except Exception as exc:
            last_error = exc
            print(
                f"Indicizzazione non pronta, tentativo {attempt}/30: {exc}",
                flush=True,
            )
            time.sleep(5)
    else:
        raise RuntimeError(f"Indicizzazione fallita: {last_error}")

    print(
        f"Indicizzazione completata: {len(documents)} file, {len(chunks)} chunk",
        flush=True,
    )
    
    # Crea file flag per indicare che l'indicizzazione è completata
    flag_file = BASE_DIR / ".index_complete"
    flag_file.touch()
    print(f"Creato flag file: {flag_file}", flush=True)


if __name__ == "__main__":
    build_index()
