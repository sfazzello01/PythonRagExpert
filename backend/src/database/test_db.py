"""Test interattivo semplice per verificare il database e le funzioni CRUD."""

from __future__ import annotations


from datetime import datetime
from pprint import pprint


from .db_manipolation import DatabaseManipulator  # type: ignore[no-redef]
from .db_retrieve import DatabaseRetriever  # type: ignore[no-redef]
from .inizializzazione import create_tables  # type: ignore[no-redef]


db_man = DatabaseManipulator()
db_ret = DatabaseRetriever()


def _prompt(message: str, default: str) -> str:
    value = input(f"{message} [{default}]: ").strip()
    return value or default


def _prompt_float(message: str, default: float) -> float:
    while True:
        value = input(f"{message} [{default}]: ").strip()
        if not value:
            return default

        try:
            return float(value)
        except ValueError:
            print("Valore non valido, riprova.")


def _prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"

    while True:
        value = input(f"{message} {suffix}: ").strip().lower()
        if not value:
            return default
        if value in {"s", "si", "y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Risposta non valida, usa s o n.")


def _default_url() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"https://example.com/test-db-{timestamp}"


def _show_record(label: str, record: tuple | None) -> None:
    print(f"\n{label}:")
    if record is None:
        print("Nessun record trovato.")
    else:
        pprint(record)


def main() -> None:
    print("Test interattivo del database")
    print("Premi Invio per usare i valori di default.")

    if _prompt_yes_no("Vuoi creare o verificare le tabelle adesso?", True):
        create_tables()

    url = _prompt("URL di test", _default_url())
    domain = _prompt("Dominio", "example.com")
    title = _prompt("Titolo", "Test database")
    html_text = _prompt(
        "HTML",
        "<html><body><p>Test HTML</p></body></html>",
    )
    gold_text = _prompt("Gold standard", "Testo gold standard di esempio.")
    parsed_text = _prompt("Testo parsato", "Testo parsato di esempio.")
    precision = _prompt_float("Precisione", 1.0)
    recall = _prompt_float("Recall", 1.0)
    f1_score = _prompt_float("F1 score", 1.0)
    lev_dist_norm = _prompt_float("Levenshtein normalizzata", 0.0)
    jac_sim_norm = _prompt_float("Jaccard normalizzata", 1.0)
    llm_score = _prompt_float("LLM score", 1.0)
    llm_comment = _prompt("Commento LLM", "Commento di test.")

    print("\nInserimento dei dati...")
    db_man.insert_web_resource(url, domain, title, html_text)
    db_man.insert_gold_standard(url, gold_text)
    db_man.insert_parsed_text(url, parsed_text)
    db_man.insert_evaluation(
        url, precision, recall, f1_score, lev_dist_norm, jac_sim_norm
    )
    db_man.insert_llm_evaluation(url, llm_score, llm_comment)

    print("\nVerifica dei dati salvati...")
    _show_record("web_resources", db_ret.get_web_resource(url))
    _show_record("gold_standard", db_ret.get_gold_standard(url))
    _show_record("parsed_text", db_ret.get_parsed_text(url))
    _show_record("evaluation", db_ret.get_evaluation(url))
    _show_record("llm_eval", db_ret.get_llm_eval(url))

    if _prompt_yes_no("\nVuoi eliminare i dati di test ora?", True):
        db_man.delete_llm_evaluation(url)
        db_man.delete_evaluation(url)
        db_man.delete_parsed_text(url)
        db_man.delete_gold_standard(url)
        db_man.delete_web_resource(url)
        print("Pulizia completata.")
    else:
        print("I dati di test sono stati lasciati nel database.")


if __name__ == "__main__":
    main()
