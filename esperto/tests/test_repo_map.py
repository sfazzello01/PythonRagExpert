"""Unit test per la mappa del repository inserita nel prompt."""

from langchain_core.documents import Document

from src.core.indexing import repo_map


def _document(filename: str, chunk_type: str, symbol: str) -> Document:
    return Document(page_content="", metadata={"filename": filename, "chunk_type": chunk_type, "symbol_name": symbol})


DOCUMENTS = [
    _document("tests/test_api.py", "function", "test_get"),
    _document("src/api.py", "function", "get"),
    _document("src/api.py", "class", "Session"),
    _document("src/api.py", "method", "Session.send"),
    _document("docs/guide.md", "text_block", "part_1"),
]


def test_map_lists_files_with_top_level_symbols_and_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text('# Progetto\n<img src="logo.png" />\nFa una [cosa](http://x) davvero utile.\n', encoding="utf-8")
    nested = tmp_path / "docs" / "README.md"
    nested.parent.mkdir()
    nested.write_text("# Documentazione\n", encoding="utf-8")

    text = repo_map.build_repo_map(DOCUMENTS, [nested, readme])

    assert "Fa una cosa davvero utile." in text and "logo" not in text and "Documentazione" not in text
    listing = text.split("FILE E SIMBOLI PRINCIPALI:\n")[1].splitlines()
    # Metodi esclusi, file senza simboli elencati per percorso, test riassunti per cartella.
    assert listing == ["docs/guide.md", "src/api.py: get, Session", "tests/ (1 file di test)"]


def test_map_drops_symbols_then_files_to_fit_the_budget(monkeypatch):
    monkeypatch.setattr(repo_map, "REPO_MAP_MAX_CHARS", 50)
    listing = repo_map.build_repo_map(DOCUMENTS, []).split("\n")[1:]
    assert listing == ["docs/guide.md", "src/api.py", "tests/ (1 file di test)"]

    monkeypatch.setattr(repo_map, "REPO_MAP_MAX_CHARS", 30)
    listing = repo_map.build_repo_map(DOCUMENTS, []).split("\n")[1:]
    assert listing == ["docs/guide.md", "src/api.py", "... (+1 altri)"]
