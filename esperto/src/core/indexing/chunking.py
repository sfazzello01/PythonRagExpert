"""Chunking semantico dei file Python tramite AST e suddivisione testuale degli altri file."""

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
# indicizzata come intestazione (con le firme dei metodi) + un chunk per metodo.
CLASS_SPLIT_MIN_LINES = 400

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _methods(node: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Elenca i metodi definiti direttamente nel corpo di una classe.

    Args:
        node: Definizione della classe.

    Returns:
        I metodi, sincroni e asincroni, nell'ordine del sorgente.
    """
    return [child for child in node.body if isinstance(child, _FUNCTION_NODES)]


def splits_into_methods(node: ast.AST) -> bool:
    """Indica se una classe è indicizzata come intestazione più un chunk per metodo.

    Args:
        node: Nodo AST da valutare.

    Returns:
        True se il nodo è una classe più lunga di CLASS_SPLIT_MIN_LINES righe con almeno un metodo.
    """
    return (
        isinstance(node, ast.ClassDef)
        and node.end_lineno - node.lineno + 1 > CLASS_SPLIT_MIN_LINES
        and bool(_methods(node))
    )


def symbol_chunk_id(relative_path: str, qualified_name: str, node: ast.AST) -> str:
    """Calcola l'ID stabile del chunk di un simbolo, condiviso con il symbol graph.

    Args:
        relative_path: Percorso del file relativo alla radice della codebase.
        qualified_name: Nome del simbolo (`Classe.metodo` per i metodi).
        node: Definizione del simbolo, di cui si usano la riga di `def`/`class` e quella finale.

    Returns:
        L'hash MD5 esadecimale che identifica il chunk.
    """
    return hashlib.md5(
        f"{relative_path}::{qualified_name}::{node.lineno - 1}::{node.end_lineno}".encode("utf-8")
    ).hexdigest()


def _first_line(node: ast.AST) -> int:
    """Trova la prima riga di una definizione, decoratori compresi.

    Args:
        node: Definizione di funzione o classe.

    Returns:
        L'indice della riga, contato da zero.
    """
    return min((decorator.lineno for decorator in node.decorator_list), default=node.lineno) - 1


def fallback_text_split(path: Path, text: str) -> list[Document]:
    """Divide un file in blocchi di testo, conservandone i metadati di origine.

    È usata per i file non Python e per quelli Python con errori di sintassi.

    Args:
        path: Percorso del file.
        text: Contenuto del file.

    Returns:
        Un documento per blocco, con file, servizio e ID del chunk nei metadati.
    """
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
    """Crea i chunk di funzioni e classi di un file Python, arricchiti con il contesto del modulo."""

    def __init__(self, file_path: Path, code: str):
        """Prepara il chunker per un file.

        Args:
            file_path: Percorso del file Python.
            code: Sorgente del file.
        """
        self.file_path = file_path
        self.relative_path = relative_path(file_path)
        self.code = code
        self.lines = code.splitlines(keepends=True)
        self.imports_context = ""

    def extract_chunks(self) -> list[Document]:
        """Estrae i chunk semantici del file.

        Returns:
            Un documento per ogni funzione o classe di primo livello (le classi lunghe sono spezzate in
            intestazione e metodi) e uno per il codice globale; con un errore di sintassi, i blocchi di
            testo prodotti da `fallback_text_split`.
        """
        try:
            tree = ast.parse(self.code)
        except SyntaxError as error:
            logger.warning("Errore di sintassi in %s: %s; uso la suddivisione testuale.", self.relative_path, error)
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
        """Estrae il sorgente di una definizione, decoratori compresi.

        Args:
            node: Definizione di funzione, metodo o classe.

        Returns:
            Le righe del file che contengono la definizione.
        """
        return "".join(self.lines[_first_line(node) : node.end_lineno])

    def _split_class(self, node: ast.ClassDef) -> list[Document]:
        """Spezza una classe lunga in un chunk di intestazione e un chunk per metodo.

        L'intestazione contiene firma, docstring e attributi della classe, più le firme dei metodi.

        Args:
            node: Definizione della classe.

        Returns:
            Il documento dell'intestazione seguito da quelli dei metodi.
        """
        methods = _methods(node)
        method_lines = {
            index for method in methods for index in range(_first_line(method), method.end_lineno)
        }
        header = "".join(
            line
            for index, line in enumerate(self.lines[: node.end_lineno])
            if index >= _first_line(node) and index not in method_lines and line.strip()
        )
        # Le firme dei metodi: chi legge l'intestazione vede la forma dell'intera classe.
        header += "".join(self._signature(method) for method in methods)
        return [
            self._symbol_document(node, "class", node.name, header),
            *(
                self._symbol_document(method, "method", f"{node.name}.{method.name}", self._source(method))
                for method in methods
            ),
        ]

    def _signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Estrae la firma di un metodo, senza decoratori né corpo.

        Args:
            node: Definizione del metodo.

        Returns:
            Le righe del `def` seguite da ` ...`.
        """
        lines = self.lines[node.lineno - 1 : node.body[0].lineno - 1] or [self.lines[node.lineno - 1]]
        return "".join(lines).rstrip() + " ...\n"

    def _symbol_document(self, node: ast.AST, chunk_type: str, name: str, source: str) -> Document:
        """Costruisce il chunk di una funzione, classe o metodo.

        Args:
            node: Definizione del simbolo.
            chunk_type: Tipo di chunk: `function`, `class` o `method`.
            name: Nome del simbolo (`Classe.metodo` per i metodi).
            source: Sorgente da inserire nel chunk.

        Returns:
            Il documento con intestazione, import del modulo, sorgente e metadati del simbolo.
        """
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
