"""Utility functions for web crawlers."""

from http.client import HTTPException

import requests  # type: ignore
from Levenshtein import distance  # type: ignore
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse
import mariadb  # type: ignore
import os
from typing import List
from ..database.db_retrieve import DatabaseRetriever
from ..llm_judge.judge import judge_texts  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import mistune  # type: ignore

from ..log import get_logger

from config import (
    MODEL,
    GS_DATA_DIR,
    BACKEND_GS_HTML_DIR,
    DOMAINS_FILE,
    LOGS_FILE,
    NO_LLM,
)

logger = get_logger(__name__)

DatabaseRetriever = DatabaseRetriever()


def iter_gs_files() -> list[Path]:
    """
    Restituisce tutti i file JSON del gold standard ordinati per nome.

    Args:
        Nessuno.

    Returns:
        Lista di percorsi ai file gs_*.json (list[Path]).
    """
    return sorted(GS_DATA_DIR.glob("gs_*.json"))


def load_json_dict(file_path: Path) -> dict[str, Any] | None:
    """
    Carica un file JSON e restituisce il contenuto se e un dizionario valido.

    Args:
        file_path: Percorso del file JSON da caricare (Path).

    Returns:
        Dizionario con i dati del file oppure None in caso di errore o formato non valido (dict[str, Any] | None).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def iter_gs_entries() -> Iterator[tuple[Path, dict[str, Any]]]:
    """
    Itera su tutte le entry valide presenti nei file del gold standard.

    Args:
        Nessuno.

    Yields:
        Iteratore di tuple contenenti file sorgente ed entry del gold standard (Iterator[tuple[Path, dict[str, Any]]]).
    """
    for gs_file in iter_gs_files():
        data = load_json_dict(gs_file)
        if data is None:
            continue

        for item in _extract_gs_items(data):
            if isinstance(item, dict):
                yield gs_file, item


def _extract_gs_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Estrae la lista di entry GS JSON.

    Args:
        data: Dizionario caricato dal file GS (dict[str, Any]).

    Returns:
        Lista di entry GS valide (list[dict[str, Any]]).
    """

    if isinstance(data.get("gold_standard"), list):
        return [item for item in data["gold_standard"] if isinstance(item, dict)]

    return []


def url_exists_in_gs(url: str) -> bool:
    """
    Verifica se un URL e gia presente nei file gs_*.json.

    Args:
        url: URL da cercare (str).

    Returns:
        True se l'URL e presente, altrimenti False (bool).
    """
    for _, item in iter_gs_entries():
        if item.get("url") == url:
            return True

    return False


def get_saved_gs_html_path(url: str) -> str | None:
    """
    Restituisce il percorso locale del file HTML GS se gia salvato in backend/gs_html.

    Args:
        url: URL da cercare nei dati del gold standard (str).

    Returns:
        Percorso del file locale se esiste, altrimenti None (str | None).
    """
    target_dir = BACKEND_GS_HTML_DIR

    if not target_dir.exists():
        return None

    for gs_file in iter_gs_files():
        data = load_json_dict(gs_file)
        if data is None:
            continue

        gold_standard_list = data.get("gold_standard", [])
        if not isinstance(gold_standard_list, list):
            continue

        for idx, item in enumerate(gold_standard_list, 1):
            if not isinstance(item, dict):
                continue
            if item.get("url") != url:
                continue

            title = str(item.get("title", f"gold_standard_{idx}"))
            safe_title = file_name_sanitize(title)
            output_filename = f"{gs_file.stem}_{safe_title}.html"
            output_path = target_dir / output_filename

            if output_path.exists():
                return str(output_path)

    return None


def file_name_sanitize(name: str) -> str:
    """
    Sanitizza una stringa per poterla usare come nome file.

    Args:
        name: Nome originale da normalizzare (str).

    Returns:
        Nome file sanitizzato (str).
    """
    return (
        "".join(c for c in name if c.isalnum() or c in (" ", "_", "-"))
        .replace(":", "")
        .replace(" ", "_")
        .rstrip()
    )


def to_string(data: Any) -> str:
    """
    Converte l'input in una stringa normalizzata, preservando spaziature e punteggiatura.

    Args:
        data: input da convertire in stringa (Any).

    Returns:
        Stringa normalizzata (str).
    """

    def _iter_blocks(value: Any) -> Iterator[str]:
        if isinstance(value, dict):
            if "text" in value:
                yield from _iter_blocks(value["text"])
                return

            for nested in value.values():
                yield from _iter_blocks(nested)
            return

        if isinstance(value, list):
            for item in value:
                yield from _iter_blocks(item)
            return

        if isinstance(value, str):
            lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
            normalized = "\n".join(line for line in lines if line)
            if normalized:
                yield normalized

    blocks = [block for block in _iter_blocks(data) if block]
    if not blocks:
        return ""

    output_parts: list[str] = []
    prev_list_item = False
    prev_quote = False

    for block in blocks:
        stripped = block.lstrip()
        is_list_item = bool(re.match(r"^([-*+]|\d+\.)\s+", stripped))
        is_quote = stripped.startswith(">")
        if output_parts:
            output_parts.append(
                "\n"
                if (prev_list_item and is_list_item) or (prev_quote and is_quote)
                else "\n\n"
            )
        output_parts.append(block)
        prev_list_item = is_list_item
        prev_quote = is_quote

    return "".join(output_parts)


def remove_citations(text: str) -> str:
    """
    Rimuove dal testo i riferimenti numerici e alfanumerici singoli
    tra parentesi quadre.

    Args:
        text: Testo da ripulire (str).

    Returns:
        Testo senza riferimenti numerici e alfanumerici singoli (str).
    """
    # Rimuove [ numero ] con eventuali spazi interni ed esterni
    return re.sub(r"\s*\[\s*\w\s*\]\s*", " ", re.sub(r"\s*\[\s*\d+\s*\]\s*", " ", text))


def save_gs_html(url: str | None = None) -> str | list[str] | None:
    """
    Salva i contenuti HTML del gold standard nella directory dedicata.

    Args:
        url: URL specifico da salvare; se None salva tutti gli HTML disponibili (str | None).

    Returns:
        Percorso del file salvato, lista dei file salvati oppure None se non e stato salvato nulla (str | list[str] | None).
    """
    source_dir = GS_DATA_DIR
    target_dir = BACKEND_GS_HTML_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    # Trova tutti i file gs_*.json
    gs_files = iter_gs_files()

    if not gs_files:
        logger.info(f"Nessun file gs_*.json trovato in {source_dir}")
        return None

    saved_paths = []
    found = False

    for gs_file in gs_files:
        data = load_json_dict(gs_file)
        if data is None:
            logger.info(f"✗ Errore di parsing JSON: {gs_file.name}")
            continue

        try:
            # Estrae la lista del gold standard
            gold_standard_list = data.get("gold_standard", [])

            for idx, item in enumerate(gold_standard_list, 1):
                # Se url è specificato, salta gli item che non corrispondono
                if url is not None and item.get("url") != url:
                    continue

                if url is not None:
                    found = True

                title = item.get("title", f"gold_standard_{idx}")
                html_text = item.get("html_text", "")

                if not html_text:
                    continue

                # Crea il nome del file dall'html
                safe_title = file_name_sanitize(title)
                output_filename = f"{gs_file.stem}_{safe_title}.html"
                output_path = target_dir / output_filename

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_text)

                logger.info(f"✓ Creato: {output_filename}")
                saved_paths.append(str(output_path))

        except Exception as e:
            logger.error(f"✗ Errore durante l'elaborazione di {gs_file.name}: {e}")

    if url is not None and not found:
        logger.error(f"✗ URL non trovato: {url}")
        return None
    elif saved_paths:
        if url is not None:
            logger.info(f"\n✓ File salvato: {saved_paths[0]}")
            return saved_paths[0]
        else:
            logger.info(
                f"\n✓ {len(saved_paths)} file HTML salvati nella cartella: {target_dir}"
            )
            return saved_paths
    else:
        logger.error("✗ Nessun file HTML salvato")
        return None


def domini_supportati() -> list[str]:
    """
    Restituisce l'elenco dei domini supportati dal parser.

    Args:
        Nessuno.

    Returns:
        Lista di domini supportati (list[str]).
    """
    data = load_json_dict(DOMAINS_FILE)
    if data is None:
        return []

    domains = data.get("domains")
    if not isinstance(domains, list):
        return []

    return [domain for domain in domains if isinstance(domain, str)]


def estrattore_dominio(url: str) -> str:
    """
    Estrae il dominio host da un URL delegando all'helper dedicato.

    Args:
        url: URL da analizzare (str).

    Returns:
        Dominio estratto in minuscolo (str).
    """
    parsed = urlparse(url.strip())
    return parsed.hostname.lower() if parsed.hostname else ""


def remove_mrkdwn_for_valuta(testo: str) -> str:
    """
    Pulisce un testo rimuovendo riferimenti numerici e normalizzando spazi.

    Args:
        testo: Testo da pulire (str).

    Returns:
        Testo pulito e normalizzato (str).
    """
    html = mistune.html(testo)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        tag.unwrap()
    text = re.sub(r"[ \t]+", " ", str(soup))
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _tokenize_for_eval(testo: str) -> list[str]:
    return re.findall(r"\b\w+\b", testo.lower())


def valuta(
    testo_parsato: str, testo_gold: str, url: str
) -> dict[str, float] | dict[str, int]:
    """
    Calcola precision, recall e F1 confrontando token unici di due testi.

    Args:
        testo_parsato: Testo prodotto dal parser (str).
        testo_gold: Testo di riferimento gold standard (str).
        url: URL della pagina analizzata (str).

    Returns:
        Dizionario con metriche precision, recall e f1 (dict[str, float]).
    """

    testo_parsato = remove_mrkdwn_for_valuta(testo_parsato)
    testo_gold = remove_mrkdwn_for_valuta(testo_gold)

    parsed_tokens = _tokenize_for_eval(testo_parsato)
    gold_tokens = _tokenize_for_eval(testo_gold)

    parsed_counts = Counter(parsed_tokens)
    gold_counts = Counter(gold_tokens)

    # Calcolo intersezione (bag-of-words)
    n_intersec = sum((parsed_counts & gold_counts).values())
    n_parsato = len(parsed_tokens)
    n_gold = len(gold_tokens)

    # Precision
    precision = n_intersec / n_parsato if n_parsato > 0 else 0.0

    # Recall
    recall = n_intersec / n_gold if n_gold > 0 else 0.0

    # F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # Levenshtein
    max_len = max(len(testo_parsato), len(testo_gold))
    normalized_lev_distance = (
        1 - (distance(testo_parsato, testo_gold) / max_len) if max_len else 0.0
    )

    # Jaccard Similarity
    parsed_set = set(parsed_tokens)
    gold_set = set(gold_tokens)
    union = parsed_set.union(gold_set)
    inter = parsed_set.intersection(gold_set)
    jaccard_similarity = len(inter) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "levenshtein_distance": normalized_lev_distance,
        "jaccard_similarity": jaccard_similarity,
    }


def carica_tutte_le_entry_gs() -> list[dict]:
    """
    Carica tutte le entry del gold standard dai file disponibili.

    Args:
        Nessuno.

    Returns:
        Lista di entry del gold standard (list[dict]).
    """
    entries = []

    for _, item in iter_gs_entries():
        entries.append(item)

    return entries


def get_gs_entry_by_domain(domain: str) -> list[dict]:
    """
    Restituisce tutte le entry del gold standard relative a un dominio.

    Args:
        domain(str): Dominio da filtrare.

    Returns:
        Lista di entry appartenenti al dominio richiesto (list[dict]).
    """
    ris = []
    for item in carica_tutte_le_entry_gs():
        if item.get("domain") == domain:
            ris.append(item)
    return ris


def validate_supported_domain(domain: str) -> str:
    """
    Valida che il dominio richiesto sia tra quelli supportati.

    Args:
        domain: Dominio da verificare (str).

    Returns:
        Dominio validato (str).
    """
    if domain not in domini_supportati():
        logger.warning(f"Dominio non supportato: {domain}")
        raise HTTPException(status_code=400, detail="Dominio non supportato")
    return domain


def validate_domain_from_url(url: str) -> str:
    """
    Estrae e valida il dominio a partire da un URL.

    Args:
        url: URL da analizzare (str).

    Returns:
        Dominio estratto e validato (str).
    """

    return validate_supported_domain(estrattore_dominio(url))


def fallback_markdown(title: str, parsed_text: str) -> str:
    """
    fallback per un output Markdown esplicito: se il titolo
    non è formattato correttamente aggiunge la formattazione corretta (#).

    Args:
        title(str): Titolo del documento.
        parsed_text(str): Testo parsato del documento.

    Returns:
        Contenuto in formato Markdown in modo esplicito.
    """
    titolo = (title or "").strip()
    testo = (parsed_text or "").strip()
    if titolo.startswith("#"):
        return f"{titolo}\n\n{testo}"
    else:
        return f"# {titolo}\n\n{testo}"


def _tail_backend_log(lines: int) -> str:
    """
    Restituisce le ultime `lines` righe del file di log del backend.

    Args:
        lines: Numero di righe da restituire (int).

    Returns:
        Stringa contenente le ultime `lines` righe del log (str).
    """
    max_lines = min(max(lines, 1), 2000)
    if not LOGS_FILE.exists():
        return ""

    with LOGS_FILE.open("r", encoding="utf-8", errors="replace") as log_file:
        return "".join(log_file.readlines()[-max_lines:])


def db_connect():
    """
    Stabilisce una connessione al database MariaDB.

    Args:
        Nessuno.

    Returns:
        Connessione al database (mariadb.Connection).
    """
    return mariadb.connect(
        host=os.getenv("MYSQL_HOST", "database"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
    )


def db_available() -> bool:
    """
    Verifica se il database è disponibile.

    Args:
        Nessuno.

    Returns:
        True se il database è disponibile, False altrimenti (bool).
    """
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Database non disponibile: {e}")
        return False


def db_fetch_all(query: str, params: tuple = ()) -> List[tuple]:
    """
    Esegue una query SQL e restituisce tutte le righe risultanti.

    Args:
        query: Query SQL da eseguire (str).
        params: Parametri per la query SQL (tuple).

    Returns:
        Lista di tuple con i risultati della query (List[tuple]).
    """
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


def empty_token_eval() -> dict[str, float]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def build_parse_response(
    url: str,
    domain: str,
    result: dict[str, Any],
    fallback_html: str = "",
) -> dict[str, str]:
    """
    Costruisce una risposta per il parsing di un URL.

    Args:
        url (str): URL da analizzare.
        domain (str): Dominio dell'URL.
        result (dict[str, Any]): Dati del parsing.
        fallback_html (str): Testo HTML di fallback.

    Returns:
        Dati della risposta (dict[str, str]).
    """
    title = result.get("title", "") or ""
    parsed_text = result.get("parsed_text", "") or ""
    markdown = fallback_markdown(title, parsed_text)
    return {
        "url": result.get("url", url) or url,
        "domain": result.get("domain", domain) or domain,
        "title": title,
        "html_text": result.get("html_text", fallback_html) or fallback_html,
        "parsed_text": markdown,
    }


def count_by_domain(table_name: str) -> dict[str, int]:
    """
    Conta il numero di record per dominio in una tabella specificata.

    Args:
        table_name (str): Nome della tabella da analizzare.

    Returns:
        Dizionario con il conteggio dei record per dominio (dict[str, int]).
    """
    allowed_tables = {
        "web_resources",
        "gold_standard",
        "parsed_text",
        "evaluation",
        "llm_eval",
    }

    if table_name not in allowed_tables:
        raise ValueError("Tabella non consentita")

    if table_name == "web_resources":
        rows = db_fetch_all(
            """
            SELECT domain, COUNT(*)
            FROM web_resources
            GROUP BY domain
            ORDER BY domain
            """
        )

    else:
        rows = db_fetch_all(
            f"""
            SELECT wr.domain, COUNT(*)
            FROM {table_name} t
            JOIN web_resources wr ON wr.url = t.url
            GROUP BY wr.domain
            ORDER BY wr.domain
            """
        )

    return {str(row[0]): int(row[1]) for row in rows}


def avg_eval_by_domain() -> dict[str, dict[str, float]]:
    """
    Calcola la media delle metriche di valutazione per dominio.

    Returns:
        Dizionario con le medie delle metriche per dominio (dict[str, dict[str, float]]).
    """
    rows = db_fetch_all(
        """
        SELECT
            wr.domain,
            AVG(ev.val_precision),
            AVG(ev.val_recall),
            AVG(ev.val_f1_score),
            AVG(ev.val_lev_dist_norm),
            AVG(ev.val_jac_sim_norm)
        FROM web_resources wr
        LEFT JOIN evaluation ev ON wr.url = ev.url
        GROUP BY wr.domain
        ORDER BY wr.domain
        """
    )

    return {
        str(row[0]): {
            "precision": round(float(row[1] or 0.0), 4),
            "recall": round(float(row[2] or 0.0), 4),
            "f1": round(float(row[3] or 0.0), 4),
            "levenshtein_distance": round(float(row[4] or 0.0), 4),
            "jaccard_similarity": round(float(row[5] or 0.0), 4),
        }
        for row in rows
    }


def avg_llm_eval_by_domain() -> dict[str, dict[str, float]]:
    """
    Calcola la media del punteggio LLM per dominio.

    Returns:
        Dizionario con la media del punteggio LLM per dominio (dict[str, dict[str, float]]).
    """
    rows = db_fetch_all(
        """
        SELECT
            wr.domain,
            AVG(le.val_llm_score)
        FROM web_resources wr
        LEFT JOIN llm_eval le ON wr.url = le.url
        GROUP BY wr.domain
        ORDER BY wr.domain
        """
    )

    return {
        str(row[0]): {"judge_score": round(float(row[1] or 0.0), 4)} for row in rows
    }


def gold_standard_from_db(url: str) -> dict[str, Any] | None:
    """
    Ricostruisce un GoldStandard completo usando:
    - web_resources: url, domain, title, html_text, created_at
    - gold_standard: url, gold_text, created_at

    Args:
        url (str): URL da cercare.

    Returns:
        Dizionario con i dati del GoldStandard o None se non trovato.
    """
    try:
        web_resource = DatabaseRetriever.get_web_resource(url)
        gold_standard = DatabaseRetriever.get_gold_standard(url)

        if web_resource is None or gold_standard is None:
            return None

        return {
            "url": web_resource[0],
            "domain": web_resource[1],
            "title": web_resource[2],
            "html_text": web_resource[3],
            "gold_text": gold_standard[1],
        }

    except Exception as e:
        logger.warning(f"Errore lettura gold standard da DB: {e}")
        return None


def full_gold_standard_from_db(domain: str) -> list[dict[str, Any]]:
    """
    Recupera tutti i GoldStandard per un dominio specifico.

    Args:
        domain (str): Dominio da cercare.

    Returns:
        Lista di dizionari con i dati dei GoldStandard.
    """
    try:
        rows = db_fetch_all(
            """
            SELECT 
                wr.url,
                wr.domain,
                wr.title,
                wr.html_text,
                gs.gold_text
            FROM gold_standard gs
            JOIN web_resources wr ON wr.url = gs.url
            WHERE wr.domain = ?
            ORDER BY gs.created_at DESC
            """,
            (domain,),
        )

        return [
            {
                "url": row[0],
                "domain": row[1],
                "title": row[2],
                "html_text": row[3],
                "gold_text": row[4],
            }
            for row in rows
        ]

    except Exception as e:
        logger.warning(f"Errore lettura full gold standard da DB: {e}")
        return []


def gold_standard_urls_from_db(domain: str) -> list[str]:
    """
    Recupera tutti gli URL dei GoldStandard per un dominio specifico.

    Args:
        domain (str): Dominio da cercare.

    Returns:
        Lista di URL dei GoldStandard.
    """
    try:
        rows = db_fetch_all(
            """
            SELECT gs.url
            FROM gold_standard gs
            JOIN web_resources wr ON wr.url = gs.url
            WHERE wr.domain = ?
            ORDER BY gs.created_at DESC
            """,
            (domain,),
        )

        return [row[0] for row in rows]

    except Exception as e:
        logger.warning(f"Errore lettura URL gold standard da DB: {e}")
        return []


def ollama_available() -> bool:
    """
    Verifica se il servizio Ollama è raggiungibile.

    Returns:
        True se Ollama risponde correttamente, altrimenti False (bool).
    """
    try:
        risposta = requests.get("http://ollama:11434", timeout=2)
        return risposta.status_code == 200
    except requests.RequestException:
        logger.warning("Ollama non è raggiungibile")
        return False


def esperto_available() -> bool:
    """
    Verifica se il servizio Esperto è raggiungibile e l'indicizzazione è completata.

    Returns:
        True se Esperto risponde correttamente e l'indicizzazione è completata, altrimenti False (bool).
    """
    try:
        risposta = requests.get("http://esperto:5000/health", timeout=5)
        logger.info(f"Esperto health check status: {risposta.status_code}")
        if risposta.status_code == 200:
            data = risposta.json()
            logger.info(f"Esperto health response: {data}")
            return data.get("status") == "ok"
        logger.warning(f"Esperto health check failed with status: {risposta.status_code}")
        return False
    except requests.RequestException as e:
        logger.warning(f"Esperto non è raggiungibile: {e}")
        return False
    except Exception as e:
        logger.error(f"Errore imprevisto in esperto_available: {e}")
        return False


async def run_judge(
    parsed_text: str, gold_text: str, url: str = "N/A"
) -> dict[str, Any]:
    """
    Esegue la valutazione LLM confrontando testo parsato e gold standard.
    
    Args:
        parsed_text (str): Testo prodotto dal parser.
        gold_text (str): Testo di riferimento gold standard.
        url (str): URL della pagina analizzata (opzionale, default "N/A").
    Returns:
        Dizionario con i risultati della valutazione LLM, inclusi punteggio e feedback (dict[str, Any]).
    """
    if NO_LLM:
        return {
            "model_name": MODEL,
            "judge_score": 1,
            "judge_feedback": "LLM evaluation saltata in modalità di testing",
        }

    raw_result = await judge_texts(
        parsed_text=parsed_text,
        gold_text=gold_text,
        model=MODEL,
        url=url,
    )

    if not isinstance(raw_result, dict):
        return {
            "model_name": MODEL,
            "judge_score": 1,
            "judge_feedback": str(raw_result),
            "x_eval": {},
        }

    score = (
        raw_result.get("judge_score")
        or raw_result.get("score")
        or raw_result.get("val_llm_score")
        or 1
    )

    feedback = (
        raw_result.get("judge_feedback")
        or raw_result.get("feedback")
        or raw_result.get("comment")
        or raw_result.get("val_llm_comment")
        or ""
    )

    try:
        score_int = float(score)
    except Exception:
        score_int = 1.0

    return {
        "model_name": str(raw_result.get("model_name", MODEL)),
        "judge_score": score_int,
        "judge_feedback": str(feedback),
        "x_eval": raw_result,
    }
