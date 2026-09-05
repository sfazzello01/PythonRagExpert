"""Unit test per il parsing delle sotto-domande del planner."""

from src.core.rag.planner import parse_subquestions


def test_parse_valid_json():
    content = '{"subquestions": ["Come funziona il retrieval?", "Come avviene il chunking?"]}'
    assert parse_subquestions(content, "Spiega la pipeline RAG") == [
        "Come funziona il retrieval?",
        "Come avviene il chunking?",
    ]


def test_parse_json_with_surrounding_text():
    content = 'Ecco il piano:\n{"subquestions": ["Come funziona il retrieval?"]}\nFine.'
    assert parse_subquestions(content, "Spiega la pipeline RAG") == [
        "Come funziona il retrieval?"
    ]


def test_invalid_json_falls_back_to_original():
    assert parse_subquestions("non e' JSON", "Domanda originale") == ["Domanda originale"]


def test_non_list_payload_falls_back_to_original():
    assert parse_subquestions('{"subquestions": "nessuna"}', "Domanda originale") == [
        "Domanda originale"
    ]


def test_deduplicates_case_insensitive_and_filters_short():
    content = '{"subquestions": ["Come funziona il retrieval?", "COME FUNZIONA IL RETRIEVAL?", "ab", 42]}'
    assert parse_subquestions(content, "Domanda originale") == ["Come funziona il retrieval?"]


def test_empty_valid_list_falls_back_to_original():
    assert parse_subquestions('{"subquestions": []}', "Domanda originale") == [
        "Domanda originale"
    ]
