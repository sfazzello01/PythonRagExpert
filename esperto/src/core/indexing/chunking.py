"""Semantic Python AST chunking and generic text fallback chunking."""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import INDEX_CHUNK_OVERLAP, INDEX_CHUNK_SIZE
from .discovery import relative_path, service_for_path

logger = logging.getLogger("EspertoChunking")

# Oltre questa lunghezza una classe eccede la finestra dell'embedding e viene
# indicizzata come intestazione + un chunk per metodo.
CLASS_SPLIT_MIN_LINES = 400

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _methods(node: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [child for child in node.body if isinstance(child, _FUNCTION_NODES)]


def splits_into_methods(node: ast.AST) -> bool:
    """Return whether a class is indexed as a header chunk plus one chunk per method."""
    return (
        isinstance(node, ast.ClassDef)
        and node.end_lineno - node.lineno + 1 > CLASS_SPLIT_MIN_LINES
        and bool(_methods(node))
    )


def symbol_chunk_id(relative_path: str, qualified_name: str, node: ast.AST) -> str:
    """Stable chunk ID shared with the symbol graph, anchored to the def/class line."""
    return hashlib.md5(
        f"{relative_path}::{qualified_name}::{node.lineno - 1}::{node.end_lineno}".encode("utf-8")
    ).hexdigest()


def _first_line(node: ast.AST) -> int:
    """Zero-based first source line of a definition, decorators included."""
    return min((decorator.lineno for decorator in node.decorator_list), default=node.lineno) - 1


def fallback_text_split(path: Path, text: str) -> list[Document]:
    """Chunk a non-Python file while preserving its source metadata."""
    filename = relative_path(path)
    service = service_for_path(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=INDEX_CHUNK_SIZE,
        chunk_overlap=INDEX_CHUNK_OVERLAP,
    ).split_text(text)
    return [
        Document(
            page_content=(
                f"# FILE: {filename} (Part {index + 1}/{len(chunks)}) "
                f"| SERVIZIO: {service}\n\n{chunk}"
            ),
            metadata={
                "filename": filename,
                "service": service,
                "chunk_type": "text_block",
                "symbol_name": f"part_{index + 1}",
                "chunk_id": hashlib.md5(
                    f"{filename}::part_{index}".encode("utf-8")
                ).hexdigest(),
            },
        )
        for index, chunk in enumerate(chunks)
    ]


class PythonASTContextualChunker:
    """Create function and class chunks enriched with module-level context."""

    def __init__(self, file_path: Path, code: str):
        self.file_path = file_path
        self.relative_path = relative_path(file_path)
        self.code = code
        self.lines = code.splitlines(keepends=True)
        self.imports_context = ""

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
        self.imports_context = "\n".join(imports) + "\n\n" if imports else ""
        processed_lines: set[int] = set()
        documents: list[Document] = []

        for node in tree.body:
            if not isinstance(node, (*_FUNCTION_NODES, ast.ClassDef)):
                continue
            processed_lines.update(range(_first_line(node), node.end_lineno))
            if splits_into_methods(node):
                documents.extend(self._split_class(node))
                continue
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            documents.append(self._symbol_document(node, chunk_type, node.name, self._source(node)))

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

    def _source(self, node: ast.AST) -> str:
        return "".join(self.lines[_first_line(node) : node.end_lineno])

    def _split_class(self, node: ast.ClassDef) -> list[Document]:
        """Chunk a large class as its header (signature, docstring, attributes) plus its methods."""
        methods = _methods(node)
        method_lines = {
            index for method in methods for index in range(_first_line(method), method.end_lineno)
        }
        header = "".join(
            line
            for index, line in enumerate(self.lines[: node.end_lineno])
            if index >= _first_line(node) and index not in method_lines and line.strip()
        )
        return [
            self._symbol_document(node, "class", node.name, header),
            *(
                self._symbol_document(method, "method", f"{node.name}.{method.name}", self._source(method))
                for method in methods
            ),
        ]

    def _symbol_document(self, node: ast.AST, chunk_type: str, name: str, source: str) -> Document:
        """Build the chunk of a function, class, or method (`name` is `Class.method` for methods)."""
        service = service_for_path(self.file_path)
        route_metadata = ""
        for decorator in node.decorator_list:
            decorator_text = ast.unparse(decorator)
            if any(method in decorator_text for method in ("get(", "post(", "put(", "delete(", "patch(")):
                route_metadata = f"# ENDPOINT API: {decorator_text}\n"
        return Document(
            page_content=(
                f"# FILE: {self.relative_path} | SERVIZIO: {service}\n"
                f"# TIPO: {chunk_type.upper()} | NOME: {name}\n"
                f"{route_metadata}"
                f"--- IMPORTS DI CONTESTO ---\n{self.imports_context}"
                f"--- CODICE SORGENTE ---\n"
                f"{source}"
            ),
            metadata={
                "filename": self.relative_path,
                "service": service,
                "chunk_type": chunk_type,
                "symbol_name": name,
                "start_line": _first_line(node) + 1,
                "end_line": node.end_lineno,
                "chunk_id": symbol_chunk_id(self.relative_path, name, node),
            },
        )