"""Semantic Python AST chunking and generic text fallback chunking."""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..shared.settings import BASE_DIR, INDEX_CHUNK_OVERLAP, INDEX_CHUNK_SIZE
from .discovery import service_for_path

logger = logging.getLogger("EspertoChunking")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def fallback_text_split(path: Path, text: str) -> list[Document]:
    """Chunk a non-Python file while preserving its source metadata."""
    relative_path = _relative_path(path)
    service = service_for_path(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=INDEX_CHUNK_SIZE,
        chunk_overlap=INDEX_CHUNK_OVERLAP,
    ).split_text(text)
    return [
        Document(
            page_content=(
                f"# FILE: {relative_path} (Part {index + 1}/{len(chunks)}) "
                f"| SERVIZIO: {service}\n\n{chunk}"
            ),
            metadata={
                "filename": relative_path,
                "service": service,
                "chunk_type": "text_block",
                "symbol_name": f"part_{index + 1}",
                "chunk_id": hashlib.md5(
                    f"{relative_path}::part_{index}".encode("utf-8")
                ).hexdigest(),
            },
        )
        for index, chunk in enumerate(chunks)
    ]


class PythonASTContextualChunker:
    """Create function and class chunks enriched with module-level context."""

    def __init__(self, file_path: Path, code: str):
        self.file_path = file_path
        self.relative_path = _relative_path(file_path)
        self.code = code
        self.lines = code.splitlines(keepends=True)

    def extract_chunks(self) -> list[Document]:
        """Return semantic chunks for top-level Python symbols and global code."""
        try:
            tree = ast.parse(self.code)
        except SyntaxError as error:
            logger.warning("Syntax error in %s: %s; using text splitter.", self.relative_path, error)
            return fallback_text_split(self.file_path, self.code)

        imports = [
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imports_context = "\n".join(imports) + "\n\n" if imports else ""
        processed_lines: set[int] = set()
        documents: list[Document] = []

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno - 1
            end = node.end_lineno
            processed_lines.update(range(start, end))
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            route_metadata = ""
            for decorator in node.decorator_list:
                decorator_text = ast.unparse(decorator)
                if any(method in decorator_text for method in ("get(", "post(", "put(", "delete(", "patch(")):
                    route_metadata = f"# ENDPOINT API: {decorator_text}\n"

            content = (
                f"# FILE: {self.relative_path} | SERVIZIO: {service_for_path(self.file_path)}\n"
                f"# TIPO: {chunk_type.upper()} | NOME: {node.name}\n"
                f"{route_metadata}"
                f"--- IMPORTS DI CONTESTO ---\n{imports_context}"
                f"--- CODICE SORGENTE ---\n"
                f"{''.join(self.lines[start:end])}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "filename": self.relative_path,
                        "service": service_for_path(self.file_path),
                        "chunk_type": chunk_type,
                        "symbol_name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "chunk_id": hashlib.md5(
                            f"{self.relative_path}::{node.name}::{start}::{end}".encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )

        global_body = "".join(
            line for index, line in enumerate(self.lines) if index not in processed_lines and line.strip()
        )
        if len(global_body.strip()) > 50:
            documents.append(
                Document(
                    page_content=f"# FILE: {self.relative_path} (Global Body)\n\n{global_body}",
                    metadata={
                        "filename": self.relative_path,
                        "service": service_for_path(self.file_path),
                        "chunk_type": "global_body",
                        "symbol_name": "main_body",
                        "chunk_id": hashlib.md5(
                            f"{self.relative_path}::global_body".encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )
        return documents