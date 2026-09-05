"""Unit test per il chunking AST dei file Python e il fallback testuale."""

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
