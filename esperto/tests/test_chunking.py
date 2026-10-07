"""Unit test per il chunking AST dei file Python e il fallback testuale."""

from src.core.indexing import chunking
from src.core.indexing.chunking import PythonASTContextualChunker

SAMPLE_CODE = '''import os
from pathlib import Path

from fastapi import FastAPI

app = FastAPI()


@app.get("/items")
def list_items():
    """Return all items."""
    return []


class ItemService:
    """Business logic for items."""

    def get(self, item_id: int) -> dict:
        return {"id": item_id}
'''


def test_ast_chunker_extracts_function_and_class(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE_CODE, encoding="utf-8")

    documents = PythonASTContextualChunker(path, SAMPLE_CODE).extract_chunks()

    by_symbol = {doc.metadata["symbol_name"]: doc for doc in documents}
    assert "list_items" in by_symbol
    assert "ItemService" in by_symbol

    function_doc = by_symbol["list_items"]
    assert function_doc.metadata["chunk_type"] == "function"
    # Il decoratore FastAPI e' evidenziato nel chunk come endpoint API.
    assert "# ENDPOINT API:" in function_doc.page_content
    # Il decoratore fa parte del sorgente del chunk, non del corpo globale.
    assert '@app.get("/items")\ndef list_items' in function_doc.page_content
    # Ogni chunk include gli imports di contesto del modulo.
    assert "from fastapi import FastAPI" in function_doc.page_content
    assert function_doc.metadata["start_line"] <= function_doc.metadata["end_line"]

    assert by_symbol["ItemService"].metadata["chunk_type"] == "class"


def test_ast_chunker_keeps_global_body(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE_CODE, encoding="utf-8")

    documents = PythonASTContextualChunker(path, SAMPLE_CODE).extract_chunks()

    global_docs = [doc for doc in documents if doc.metadata["chunk_type"] == "global_body"]
    assert len(global_docs) == 1
    assert "app = FastAPI()" in global_docs[0].page_content


def test_syntax_error_falls_back_to_text_split(tmp_path):
    code = "def broken(:\n    pass\n"
    path = tmp_path / "broken.py"
    path.write_text(code, encoding="utf-8")

    documents = PythonASTContextualChunker(path, code).extract_chunks()

    assert documents
    assert all(doc.metadata["chunk_type"] == "text_block" for doc in documents)


def test_large_class_is_split_into_header_and_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "CLASS_SPLIT_MIN_LINES", 60)
    methods = "".join(
        f"    @property\n    def m{index}(self):\n" + "        pass\n" * 20 for index in range(3)
    )
    code = f'class Big:\n    """Docstring."""\n\n    limit = 3\n\n{methods}'
    path = tmp_path / "big.py"
    path.write_text(code, encoding="utf-8")

    documents = PythonASTContextualChunker(path, code).extract_chunks()

    by_symbol = {doc.metadata["symbol_name"]: doc for doc in documents}
    assert list(by_symbol) == ["Big", "Big.m0", "Big.m1", "Big.m2"]
    assert "limit = 3" in by_symbol["Big"].page_content
    assert "def m0" not in by_symbol["Big"].page_content
    assert by_symbol["Big.m1"].metadata["chunk_type"] == "method"
    assert "@property\n    def m1" in by_symbol["Big.m1"].page_content
    assert len({doc.metadata["chunk_id"] for doc in documents}) == 4


def test_relation_headers_are_prepended_to_the_chunk():
    from langchain_core.documents import Document

    from src.core.indexing.pipeline import _add_relation_headers

    documents = [
        Document(page_content="class Child", metadata={"chunk_id": "child"}),
        Document(page_content="def plain", metadata={"chunk_id": "plain"}),
    ]
    _add_relation_headers(documents, {"child": {"inherits": ["Base"], "calls": ["helper", "Base.run"]}})
    assert documents[0].page_content == "# EREDITA DA: Base\n# CHIAMA: helper, Base.run\nclass Child"
    assert documents[1].page_content == "def plain"
