from __future__ import annotations
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Query  # type: ignore

from dotenv import load_dotenv
from ..log.logcon import get_logger
from typing import Any

from .api_classi import (
    Domains,
    Parse,
    GoldStandard,
    FullGoldStandard,
    FullGsEval,
    EvaluateRequest,
    EvaluateResponse,
    AddGoldStandardRequest,
    AddWebResourceRequest,
    EvaluateJudgeResponce,
    GoldStandardUrls,
    ParceRequest,
    UrlRequest,
    StatusResponce,
)

from ..utilities.utilities import (
    domini_supportati,
    validate_domain_from_url,
    get_gs_entry_by_domain,
    validate_supported_domain,
    valuta,
    carica_tutte_le_entry_gs,
    _tail_backend_log,
    db_available,
    empty_token_eval,
    build_parse_response,
    count_by_domain,
    avg_eval_by_domain,
    avg_llm_eval_by_domain,
    gold_standard_from_db,
    full_gold_standard_from_db,
    gold_standard_urls_from_db,
    ollama_available,
    esperto_available,
    run_judge,
)
from ..crawler import crawl_url, parse_html_text

from ..database.db_manipolation import DatabaseManipulator
from ..database.db_retrieve import DatabaseRetriever

DatabaseManipulator = DatabaseManipulator()
DatabaseRetriever = DatabaseRetriever()

load_dotenv()
app = FastAPI(title="API Progetto")
logger = get_logger(__name__)


@app.on_event("startup")
def startup_event() -> None:
    try:
        # create_tables() #c'è già un entrypoint che avvia il db
        # seed_mariadb()
        logger.info("Tabelle database verificate correttamente")
    except Exception as e:
        logger.warning(f"Impossibile verificare/creare le tabelle DB: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:
        duration_ms = (perf_counter() - start_time) * 1000
        logger.error(
            f"{request.method} {request.url.path} - error - {duration_ms:.2f} ms - {e}"
        )
        raise

    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f} ms"
    )
    return response


# get e post


@app.get("/domains", response_model=Domains)
def get_domains() -> Domains:
    """
    Restituisce la lista dei domini supportati dal backend.

    Args:
        Nessuno.

    Returns:
        Oggetto con la lista dei domini supportati (Domains).
    """
    return {"domains": domini_supportati()}


@app.get("/logs")
def get_logs(lines: int = 200) -> dict[str, str]:
    """
    Restituisce le ultime righe del log backend.

    Args:
        lines: Numero massimo di righe da restituire (int).

    Returns:
        Testo del log backend in coda (dict[str, str]).
    """
    try:
        return {"log_text": _tail_backend_log(lines)}
    except Exception as e:
        logger.error(f"Errore lettura log backend: {e}")
        raise HTTPException(status_code=500, detail="Errore lettura log backend")

@app.post("/parse", response_model=Parse)
async def post_parse(richiesta: ParceRequest) -> Parse:
    """
    Esegue il parsing di HTML fornito nel body della richiesta.

    Args:
        richiesta: Payload con URL e contenuto HTML da parsare (ParseHtmlRequest).

    Returns:
        Dati parse del contenuto HTML inviato (Parse).
    """
    dominio = validate_domain_from_url(richiesta.url)

    try:
        if richiesta.html_text:
            result = await parse_html_text(richiesta.url, richiesta.html_text)

            response = build_parse_response(
                richiesta.url,
                dominio,
                result,
                fallback_html=richiesta.html_text,
            )

            try:
                DatabaseManipulator.insert_web_resource(
                    url=richiesta.url,
                    domain=dominio,
                    title=result.get("title", ""),
                    html_text=richiesta.html_text,
                )
                DatabaseManipulator.insert_parsed_text(
                    url=richiesta.url,
                    parsed_text=response["parsed_text"],
                )
            except Exception as e:
                logger.warning(
                    f"Salvataggio web_resource o parsed_text fallito per {richiesta.url}: {e}"
                )

            return response

        if richiesta.local:
            web_resource = DatabaseRetriever.get_web_resource(richiesta.url)

            if web_resource is None:
                raise HTTPException(
                    status_code=404,
                    detail="Risorsa web non presente nel database",
                )

            html_text = web_resource[3] or ""
            result = await parse_html_text(richiesta.url, html_text)

            response = build_parse_response(
                richiesta.url,
                dominio,
                result,
                fallback_html=html_text,
            )
            try:
                DatabaseManipulator.insert_parsed_text(
                    richiesta.url, response["parsed_text"]
                )
            except Exception as e:
                logger.warning(f"Salvataggio parsed_text fallito: {e}")

            return response

        result = await crawl_url(richiesta.url)
        response = build_parse_response(richiesta.url, dominio, result)

        try:
            saved_url = response["url"]

            DatabaseManipulator.insert_web_resource(
                url=saved_url,
                domain=response["domain"],
                title=response["title"],
                html_text=response["html_text"],
            )

            DatabaseManipulator.insert_parsed_text(
                saved_url,
                response["parsed_text"],
            )
                        
        except Exception as e:
            logger.warning(
                f"Salvataggio web_resource o parsed_text fallito per {richiesta.url}: {e}"
            )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error parsing HTML: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during parsing: {e}")
        raise HTTPException(status_code=500, detail="Errore durante il parsing")


@app.get("/gold_standard", response_model=GoldStandard)
def get_gold_standard(url: str = Query(...)) -> GoldStandard:

    """
     Restituisce il gold standard associato ad uno specifico url

     Args:
        url: URL della risorsa da cui recuperare il gold standard

     Reutrns:
        Oggetto contenente URL, dominio,titolo, html originale e testo gold standard
     
    Raises:
        HTTPExcepion: se l'url non è presente nel gold standard

    """

    gold_standard = gold_standard_from_db(url)

    if gold_standard is None:
        raise HTTPException(
            status_code=404,
            detail="URL not found in gold standard",
        )

    return {
        "url": gold_standard.get("url") or url,
        "domain": gold_standard.get("domain") or "",
        "title": gold_standard.get("title") or "",
        "html_text": gold_standard.get("html_text") or "",
        "gold_text": gold_standard.get("gold_text") or "",
    }


@app.get("/full_gold_standard", response_model=FullGoldStandard)
def get_full_gold_standard(domain: str = Query(...)) -> FullGoldStandard:
    """
    Restituisce tutte le entry di gold standard per un dominio.

    Args:
        domain: Dominio da filtrare (str).

    Returns:
        Collezione di entry del gold standard per dominio (FullGoldStandard).
    """
    validate_supported_domain(domain)

    full_gs = full_gold_standard_from_db(domain)

    if not full_gs:
        full_gs = get_gs_entry_by_domain(domain)
    if full_gs == []:
        logger.error(f"No gold standard entries found for domain: {domain}")
        raise HTTPException(status_code=404, detail="URL irraggiungibile")
    return {"gold_standard": full_gs}


@app.get("/gold_standard_urls", response_model=GoldStandardUrls)
def get_gold_standard_urls(domain: str = Query(...)) -> GoldStandardUrls:

    """
    Restituisce la lista degli URL presenti nel gold standard per un dominio.

    Args:
        domain: Dominio per cui recuperare gli URL del gold standard.

    Returns:
        Oggetto contenente la lista degli URL disponibili per il dominio richiesto.

    Raises:
        HTTPException: se il dominio non è supportato.
    """

    validate_supported_domain(domain)

    urls = gold_standard_urls_from_db(domain)

    if not urls:
        entries = get_gs_entry_by_domain(domain)
        urls = [
            entry["url"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("url"), str)
        ]

    return {"gold_standard_urls": urls}


@app.post("/evaluate_judge", response_model=EvaluateJudgeResponce)
async def post_evaluate_judge(request: EvaluateRequest) -> EvaluateJudgeResponce:

    """
    Valuta tramite LLM la qualità del testo parsato rispetto al gold standard.

    Args:
        request: richiesta contenente testo parsato, gold text e URL della risorsa.

    Returns:
        Oggetto contenente nome del modello, punteggio assegnato dal judge e feedback testuale.

    Raises:
        HTTPException: se il judge LLM non è disponibile o fallisce durante la valutazione.
    """

    try:
        url = request.url or "N/A"
        llm = await run_judge(request.parsed_text, request.gold_text, url)
        try:
            if request.url:
                DatabaseManipulator.insert_llm_evaluation(
                    request.url,
                    llm["model_name"],
                    llm["judge_score"],
                    llm["judge_feedback"],
                )
        except Exception as e:
            logger.warning(f"Salvataggio llm_eval fallito per {request.url}: {e}")
        return llm
    except Exception as e:
        logger.error(f"Errore evaluate_judge: {e}")
        raise HTTPException(
            status_code=503,
            detail="Judge LLM non disponibile o errore durante la valutazione",
        )


@app.get("/full_gs_eval", response_model=FullGsEval)
async def get_full_gs_eval(domain: str) -> FullGsEval:
    """
    Calcola le metriche medie sul gold standard per un dominio.

    Args:
        domain: Dominio su cui eseguire la valutazione (str).

    Returns:
        Metriche medie token-level per il dominio richiesto (FullGsEval).
    """
    validate_supported_domain(domain)

    entries = full_gold_standard_from_db(domain)

    if not entries:
        entries = get_gs_entry_by_domain(domain)
    if not entries:
        return {
            "token_level_eval": empty_token_eval(),
            "levenshtein_distance": 0.0,
            "jaccard_similarity": 0.0,
            "judge_score": 0.0,
            "x_eval": {
                "evaluated_entries": 0,
            },
        }

    precisions = []
    recalls = []
    f1s = []
    lev_distances = []
    jaccard_similarities = []
    llm_scores = []

    for entry in entries:
        try:
            url = entry.get("url", "")
            html_text = entry.get("html_text", "")
            gold_text = entry.get("gold_text", "")

            if not url or not gold_text:
                continue

            entry_domain = validate_domain_from_url(url)

            if html_text:
                parsed = await parse_html_text(url, html_text)
            else:
                parsed = await crawl_url(url)

            parse_response = build_parse_response(
                url,
                entry_domain,
                parsed,
                fallback_html=html_text,
            )

            parsed_text = parse_response["parsed_text"]
            metriche = valuta(parsed_text, gold_text, url)

            precisions.append(metriche["precision"])
            recalls.append(metriche["recall"])
            f1s.append(metriche["f1"])
            lev_distances.append(metriche["levenshtein_distance"])
            jaccard_similarities.append(metriche["jaccard_similarity"])

            llm_result = await run_judge(parsed_text, gold_text, url)
            llm_scores.append(llm_result["judge_score"])

            try:
                DatabaseManipulator.insert_web_resource(
                    url=url,
                    domain=entry_domain,
                    title=entry.get("title", ""),
                    html_text=entry.get("html_text", ""),
                )
                DatabaseManipulator.insert_parsed_text(
                    url=url,
                    parsed_text=parsed_text,
                )
                DatabaseManipulator.insert_evaluation(
                    url=url,
                    val_precision=metriche["precision"],
                    val_recall=metriche["recall"],
                    val_f1_score=metriche["f1"],
                    val_lev_dist_norm=metriche["levenshtein_distance"],
                    val_jac_sim_norm=metriche["jaccard_similarity"],
                )
                DatabaseManipulator.insert_llm_evaluation(
                    url,
                    llm_result["model_name"],
                    llm_result["judge_score"],
                    llm_result["judge_feedback"],
                )
            except Exception as e:
                logger.warning(
                    f"Salvataggio web_resource, parsed_text o evaluation fallito per {url}: {e}"
                )

        except Exception as e:
            logger.error(f"Errore durante full_gs_eval su entry {entry}: {e}")

    if not precisions:
        return {
            "token_level_eval": empty_token_eval(),
            "levenshtein_distance": 0.0,
            "jaccard_similarity": 0.0,
            "judge_score": 0.0,
            "x_eval": {
                "evaluated_entries": 0,
            },
        }

    return {
        "token_level_eval": {
            "precision": round(sum(precisions) / len(precisions), 4),
            "recall": round(sum(recalls) / len(recalls), 4),
            "f1": round(sum(f1s) / len(f1s), 4),
        },
        "levenshtein_distance": round(sum(lev_distances) / len(lev_distances), 4),
        "jaccard_similarity": round(
            sum(jaccard_similarities) / len(jaccard_similarities),
            4,
        ),
        "judge_score": round(sum(llm_scores) / len(llm_scores), 4)
        if llm_scores
        else 0.0,
        "x_eval": {
            "evaluated_entries": len(precisions),
            "judge_evaluated_entries": len(llm_scores),
        },
    }

@app.post("/add_web_resource")
def add_web_resource(request: AddWebResourceRequest) -> dict[str, Any]:

    """
    Inserisce una nuova risorsa web nel database.

    Args:
        request: richiesta contenente URL e, opzionalmente, contenuto HTML della risorsa.

    Returns:
        Dizionario con stato dell'operazione e URL inserito.

    Raises:
        HTTPException: se si verifica un errore durante l'inserimento nel database.
    """

    try:
        url = request.url
        html_text = getattr(request, "html_text", "") or ""

        try:
            domain = validate_domain_from_url(url)
        except Exception:
            domain = ""

        DatabaseManipulator.insert_web_resource(
            url=url,
            domain=domain,
            title="",
            html_text=html_text,
        )

        return {
            "status": "ok",
            "url": url,
        }

    except Exception as e:
        logger.error(f"Errore add_web_resource: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante inserimento web resource: {e}",
        )
    
@app.post("/add_gold_standard")
def add_gold_standard(request: AddGoldStandardRequest) -> dict[str, Any]:

    """
    Inserisce un gold standard associato a una risorsa web già presente nel database.

    Args:
        request: richiesta contenente URL della risorsa e testo gold standard.

    Returns:
        Dizionario con stato dell'operazione e URL associato al gold standard.

    Raises:
        HTTPException: se la risorsa web non esiste o se l'inserimento fallisce.
    """

    try:
        url = request.url
        gold_text = getattr(request, "gold_text", "") or ""

        web_resource = DatabaseRetriever.get_web_resource(url)

        if web_resource is None:
            raise HTTPException(
                status_code=404,
                detail="Web resource non presente nel database",
            )

        DatabaseManipulator.insert_gold_standard(
            url,
            gold_text,
        )

        return {
            "status": "ok",
            "url": url,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Errore add_gold_standard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante inserimento gold standard: {e}",
        )
    except Exception as e:
        logger.error(f"Errore add_gold_standard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante inserimento gold standard: {e}",
        )
    
@app.post("/evaluate", response_model=EvaluateResponse)
def post_evaluate(richiesta: EvaluateRequest) -> EvaluateResponse:
    """
    Calcola le metriche di valutazione tra testo parsato e gold text.

    Args:
        richiesta: Payload con parsed_text, gold_text e url (EvaluateRequest).

    Returns:
        Risultato della valutazione token-level (EvaluateResponse).
    """

    url = richiesta.url or "N/A"
    valori = valuta(richiesta.parsed_text, richiesta.gold_text, url)

    try:
        if richiesta.url:
            DatabaseManipulator.insert_evaluation(
                richiesta.url,
                val_precision=valori["precision"],
                val_recall=valori["recall"],
                val_f1_score=valori["f1"],
                val_lev_dist_norm=valori["levenshtein_distance"],
                val_jac_sim_norm=valori["jaccard_similarity"],
            )
    except Exception as e:
        logger.warning(f"Salvataggio evaluation fallito per {richiesta.url}: {e}")

    return {
        "token_level_eval": {
            "precision": valori["precision"],
            "recall": valori["recall"],
            "f1": valori["f1"],
        },
        "levenshtein_distance": valori["levenshtein_distance"],
        "jaccard_similarity": valori["jaccard_similarity"],
        "x_eval": {
            "levenshtein_distance": valori["levenshtein_distance"],
            "jaccard_similarity": valori["jaccard_similarity"],
        },
    }

@app.delete("/web_resource")
async def delete_web_resource(request: Request, url: str | None = Query(None),) -> dict[str, Any]:

    """
    Elimina una risorsa web dal database.

    L'URL può essere passato come query parameter oppure nel body JSON della richiesta.

    Args:
        request: richiesta HTTP eventualmente contenente l'URL nel body.
        url: URL della risorsa web da eliminare, se passato come query parameter.

    Returns:
        Dizionario con stato dell'operazione e URL eliminato.

    Raises:
        HTTPException: se l'URL è mancante o se si verifica un errore durante l'eliminazione.
    """

    if url is None:
        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            url = None

    if not url:
        raise HTTPException(status_code=400, detail="URL mancante")

    try:
        web_resource = DatabaseRetriever.get_web_resource(url)

        if web_resource is not None:
            DatabaseManipulator.delete_web_resource(url)

        return {
            "status": "ok",
            "url": url,
        }

    except Exception as e:
        logger.error(f"Errore delete_web_resource: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante eliminazione web resource: {e}",
        )

@app.delete("/gold_standard")
async def delete_gold_standard(request: Request, url: str | None = Query(None),) -> dict[str, Any]:

    """
    Elimina un gold standard dal database.

    L'URL può essere passato come query parameter oppure nel body JSON della richiesta.

    Args:
        request: richiesta HTTP eventualmente contenente l'URL nel body.
        url: URL del gold standard da eliminare, se passato come query parameter.

    Returns:
        Dizionario con stato dell'operazione e URL eliminato.

    Raises:
        HTTPException: se l'URL è mancante o se si verifica un errore durante l'eliminazione.
    """

    if url is None:
        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            url = None

    if not url:
        raise HTTPException(status_code=400, detail="URL mancante")

    try:
        gold_standard = DatabaseRetriever.get_gold_standard(url)

        if gold_standard is not None:
            DatabaseManipulator.delete_gold_standard(url)

        return {
            "status": "ok",
            "url": url,
        }

    except Exception as e:
        logger.error(f"Errore delete_gold_standard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante eliminazione gold standard: {e}",
        )

@app.get("/db_stats")
def get_db_stats() -> dict[str, Any]:

    """
    Restituisce statistiche aggregate sul contenuto del database.

    Le statistiche includono il numero di risorse, gold standard, testi parsati,
    valutazioni automatiche e valutazioni LLM, raggruppate per dominio.

    Returns:
        Dizionario contenente conteggi e metriche medie per dominio.
    """

    try:
        return {
            "web_resources": count_by_domain("web_resources"),
            "gold_standard": count_by_domain("gold_standard"),
            "parsed_text": count_by_domain("parsed_text"),
            "evaluation": count_by_domain("evaluation"),
            "llm_eval": count_by_domain("llm_eval"),
            "avg_eval": avg_eval_by_domain(),
            "avg_eval_judge": avg_llm_eval_by_domain(),
        }

    except Exception as e:
        logger.error(f"Errore db_stats: {e}")

        entries = carica_tutte_le_entry_gs()
        counts: dict[str, int] = {}

        for entry in entries:
            domain = entry.get("domain")

            if isinstance(domain, str):
                counts[domain] = counts.get(domain, 0) + 1

        avg_eval = {
            domain: {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "levenshtein_distance": 0.0,
                "jaccard_similarity": 0.0,
            }
            for domain in counts
        }
        avg_eval_judge = {domain: {"judge_score": 1} for domain in counts}
        return {
            "web_resources": counts,
            "gold_standard": counts,
            "parsed_text": counts,
            "evaluation": counts,
            "llm_eval": counts,
            "avg_eval": avg_eval,
            "avg_eval_judge": avg_eval_judge,
            "x_eval": {
                "warning": "DB non disponibile",
            },
        }


@app.get("/db_schema")
def get_db_schema() -> dict[str, Any]:

    """
    Restituisce lo schema logico del database utilizzato dall'applicazione.

    L'endpoint descrive le principali tabelle presenti nel database, indicando
    per ciascuna i campi principali, i tipi di dato e le relazioni tra tabelle.

    Returns:
        Dizionario contenente la struttura delle tabelle web_resources,
        gold_standard, parsed_text, evaluation e llm_eval.
    """

    return {
        "web_resources": {
            "url": "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci PRIMARY KEY",
            "domain": "VARCHAR(255)",
            "title": "VARCHAR(2048)",
            "html_text": "LONGTEXT",
            "created_at": "DATETIME",
        },
        "gold_standard": {
            "url": "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci PRIMARY KEY, FK web_resources(url) ON DELETE CASCADE",
            "gold_text": "LONGTEXT",
            "created_at": "DATETIME",
        },
        "parsed_text": {
            "url": "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci PRIMARY KEY, FK web_resources(url) ON DELETE CASCADE",
            "parsed_text": "LONGTEXT",
            "created_at": "DATETIME",
        },
        "evaluation": {
            "url": "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci PRIMARY KEY, FK parsed_text(url) ON DELETE CASCADE",
            "val_precision": "FLOAT",
            "val_recall": "FLOAT",
            "val_f1_score": "FLOAT",
            "val_lev_dist_norm": "FLOAT",
            "val_jac_sim_norm": "FLOAT",
            "created_at": "DATETIME",
        },
        "llm_eval": {
            "url": "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci PRIMARY KEY, FK parsed_text(url) ON DELETE CASCADE",
            "model_name": "VARCHAR(255)",
            "val_llm_score": "FLOAT",
            "val_llm_comment": "LONGTEXT",
            "created_at": "DATETIME",
        },
    }


@app.get("/status", response_model=StatusResponce)
def get_status() -> StatusResponce:

    """
    Restituisce lo stato dei principali servizi dell'applicazione.

    Returns:
        Oggetto contenente lo stato di backend, database e servizio Ollama.
    """

    return {
        "backend": "ok",
        "database": "ok" if db_available() else "error",
        "ollama": "ok" if ollama_available() else "error",
        "esperto": "ok" if esperto_available() else "error",
    }


@app.get("/llm_eval")
def get_llm_eval(url: str = Query(...)) -> dict[str, Any]:

    """
    Recupera dal database la valutazione LLM associata a uno specifico URL.

    Args:
        url: URL della risorsa per cui recuperare la valutazione LLM.

    Returns:
        Dizionario contenente URL, nome del modello, punteggio del judge e feedback.

    Raises:
        HTTPException: se la valutazione LLM non è presente o se il recupero fallisce.
    """

    validate_domain_from_url(url)

    try:
        result = DatabaseRetriever.get_llm_eval(url)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Valutazione LLM non trovata",
            )

        return {
            "url": result[0],
            "model_name": result[1],
            "judge_score": result[2],
            "judge_feedback": result[3],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore get_llm_eval: {e}")
        raise HTTPException(
            status_code=500,
            detail="Errore durante recupero valutazione LLM",
        )
