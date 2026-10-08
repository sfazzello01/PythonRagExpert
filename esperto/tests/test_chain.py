"""Unit test per la composizione del contesto mostrato al modello."""

from src.core.rag.chain import _build_context
from src.core.rag.prompts import build_prompt


def _source(filename: str, start_line: int, body: str) -> dict:
    content = f"# FILE: {filename}\n--- IMPORTS DI CONTESTO ---\nimport os\n\n--- CODICE SORGENTE ---\n{body}"
    return {"filename": filename, "start_line": start_line, "content": content}


def test_context_is_grouped_by_file_with_imports_shown_once():
    context = _build_context([_source("a.py", 30, "def late"), _source("b.py", 1, "def other"), _source("a.py", 5, "def early")])
    # I file nell'ordine del loro chunk migliore, i chunk di un file per riga.
    assert context.index("def early") < context.index("def late") < context.index("def other")
    # Gli import restano sul primo chunk di ogni file.
    assert context.count("import os") == 2
    assert "# FILE: a.py\n--- CODICE SORGENTE ---\ndef late" in context


def test_prompt_states_the_question_before_and_after_the_context():
    text = build_prompt().invoke({"repo_map": "MAPPA\n", "context": "CODICE", "question": "Cosa fa f?"}).messages[0].content
    assert text.startswith("MAPPA\nDOMANDA A CUI RISPONDERE: Cosa fa f?")
    assert text.index("Cosa fa f?") < text.index("CODICE") < text.rindex("Cosa fa f?")
