from __future__ import annotations

from typing import Any

import requests  # type: ignore
from requests.exceptions import ChunkedEncodingError  # type: ignore
from fastapi import FastAPI, Request  # type: ignore
from fastapi.responses import (  # type: ignore
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore

from .logcon import get_logger

from config import BACKEND_URL, ESPERTO_URL, FRONTEND_TEMPLATES_DIR, LOGS_FILE

from .classi import (
    ParseFrontendRequest,
    EvaluateFrontendRequest,
    AddWebResourceFrontendRequest,
    AddGoldStandardFrontendRequest,
    ChatFrontendRequest,
    UrlFrontendRequest,
)

app = FastAPI(title="Frontend Progetto")

template = Jinja2Templates(directory=str(FRONTEND_TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str("src/static")), name="static")

logger = get_logger(__name__)


def _tail_frontend_log(lines: int) -> str:
    """
    Restituisce le ultime righe del log frontend.

    Args:
        lines: Numero massimo di righe da restituire (int).

    Returns:
        Testo del log frontend (str).
    """
    max_lines = min(max(lines, 1), 2000)
    if not LOGS_FILE.exists():
        return ""

    with LOGS_FILE.open("r", encoding="utf-8", errors="replace") as log_file:
        return "".join(log_file.readlines()[-max_lines:])


def _backend_request(
    endpoint: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
    timeout: int = 30,
    include_error_body: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        url_base = str(BACKEND_URL).rstrip("/")
        endpoint_pulito = f"/{endpoint.lstrip('/')}"
        full_url = f"{url_base}{endpoint_pulito}"

        method_upper = method.upper()

        if method_upper == "POST":
            response = requests.post(
                full_url,
                json=json_data,
                timeout=timeout,
            )

        elif method_upper == "DELETE":
            response = requests.delete(
                full_url,
                json=json_data,
                timeout=timeout,
            )

        else:
            response = requests.get(
                full_url,
                params=params,
                timeout=timeout,
            )

        if not response.ok:
            error = f"Errore backend: {response.status_code}"

            if include_error_body:
                error = f"{error} - {response.text}"

            return None, error

        return response.json(), None

    except Exception as e:
        logger.error(f"Errore di connessione al backend: {e}")
        return None, f"Errore di connessione al backend: {e}"


def _render_backend_get(
    request: Request,
    endpoint: str,
    target_template: str = "index.html",
    *,
    params: dict[str, str] | None = None,
    gs_urls: list[str] | None = None,
    include_error_body: bool = False,
    timeout: int = 30,
) -> HTMLResponse:
    data, error = _backend_request(
        endpoint,
        params=params,
        timeout=timeout,
        include_error_body=include_error_body,
    )
    return render_page(
        request,
        target_template,
        data=data,
        gs_urls=gs_urls,
        endpoint=endpoint,
        error=error,
    )


def _render_backend_post(
    request: Request,
    endpoint: str,
    json_data: dict[str, Any],
    target_template: str = "index.html",
    *,
    include_error_body: bool = False,
    timeout: int = 30,
) -> HTMLResponse:
    data, error = _backend_request(
        endpoint,
        method="POST",
        json_data=json_data,
        timeout=timeout,
        include_error_body=include_error_body,
    )
    return render_page(
        request, target_template, data=data, endpoint=endpoint, error=error
    )


def fetch_gs_urls() -> list[str]:
    gs_urls: list[str] = []
    seen_urls: set[str] = set()

    domains_data, error = _backend_request("/domains")
    if error or domains_data is None:
        return []

    domains = domains_data.get("domains", [])
    for domain in domains:
        full_gs_data, _ = _backend_request(
            "/full_gold_standard",
            params={"domain": domain},
        )
        if full_gs_data is None:
            continue

        entries = full_gs_data.get("gold_standard", [])
        for entry in entries:
            url = entry.get("url")
            if isinstance(url, str) and url and url not in seen_urls:
                seen_urls.add(url)
                gs_urls.append(url)

    return gs_urls


def fetch_domains() -> list[str]:
    domains_data, error = _backend_request("/domains")
    if error or domains_data is None:
        return []

    domains = domains_data.get("domains", [])
    return [domain for domain in domains if isinstance(domain, str) and domain]


def fetch_status() -> dict[str, str]:
    data, error = _backend_request("/status", timeout=10)

    if error or data is None:
        return {
            "backend": "error",
            "database": "error",
            "ollama": "error",
            "esperto": "error",
        }

    return {
        "backend": str(data.get("backend", "error")),
        "database": str(data.get("database", "error")),
        "ollama": str(data.get("ollama", "error")),
        "esperto": str(data.get("esperto", "error")),
    }


def render_page(
    request: Request,
    template_name: str,
    data=None,
    gs_urls: list[str] | None = None,
    domains: list[str] | None = None,
    endpoint: str | None = None,
    error: str | None = None,
):
    status_servizi = fetch_status()

    return template.TemplateResponse(
        request,
        template_name,
        {
            "request": request,
            "data": data,
            "gs_urls": gs_urls or [],
            "domains": domains or [],
            "endpoint": endpoint,
            "error": error,
            "status": status_servizi,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_page(
        request,
        "index.html",
        gs_urls=fetch_gs_urls(),
        domains=fetch_domains(),
    )


@app.get("/parser", response_class=HTMLResponse)
def view_parser(request: Request, url: str | None = None):
    return render_page(request, "parser.html", gs_urls=fetch_gs_urls())


@app.get("/gold", response_class=HTMLResponse)
def view_gold_builder(
    request: Request, url: str | None = None, domain: str | None = None
):
    return render_page(
        request,
        "gold.html",
        domains=fetch_domains(),
    )


@app.get("/stats", response_class=HTMLResponse)
def view_global_stats(request: Request, domain: str | None = None):
    domini_disponibili = fetch_domains()
    if domain:
        data, error = _backend_request(
            "/full_gs_eval", params={"domain": domain}, timeout=120
        )
        return render_page(
            request,
            "stats.html",
            data=data,
            domains=domini_disponibili,
            endpoint="/full_gs_eval",
            error=error,
        )

    return render_page(request, "stats.html", domains=domini_disponibili)

@app.get("/esperto", response_class=HTMLResponse)
def view_esperto(request: Request):
    return render_page(
        request,
        "esperto.html",
    )


@app.post("/ask")
def ask_esperto(request: ChatFrontendRequest):
    """
    Proxy streaming verso il servizio esperto.

    Il browser chiama il frontend, mentre il container frontend raggiunge
    l'esperto tramite la rete interna Docker.
    """
    message = request.message.strip()

    if not message:
        return PlainTextResponse(
            "Inserisci una domanda prima di inviare.",
            status_code=400,
        )

    try:
        esperto_url = str(ESPERTO_URL).rstrip("/")
        full_url = f"{esperto_url}/ask"
        response = requests.post(
            full_url,
            json={"message": message},
            timeout=(10, 300),
            stream=True,
        )
    except Exception as e:
        logger.error(f"Errore di connessione all'esperto: {e}")
        return PlainTextResponse(
            f"Errore di connessione all'esperto: {e}",
            status_code=502,
        )

    if not response.ok:
        error_text = response.text
        response.close()
        return PlainTextResponse(
            f"Errore esperto: {response.status_code} - {error_text}",
            status_code=502,
        )

    def stream_response():
        try:
            try:
                for chunk in response.iter_content(
                    chunk_size=None,
                    decode_unicode=True,
                ):
                    if chunk:
                        yield chunk
            except ChunkedEncodingError as e:
                logger.error(f"Stream esperto interrotto: {e}")
                yield (
                    "\n\nErrore: lo stream dell'esperto si e' interrotto prima "
                    "di completare la risposta."
                )
        finally:
            response.close()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/logs/backend")
def frontend_backend_logs(lines: int = 200):
    data, error = _backend_request(
        "/logs", params={"lines": str(lines)}, include_error_body=True
    )
    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore recupero log backend", "log_text": ""},
        )
    return JSONResponse(content={"log_text": data.get("log_text", "")})


@app.get("/logs/frontend")
def frontend_logs(lines: int = 200):
    try:
        return JSONResponse(content={"log_text": _tail_frontend_log(lines)})
    except Exception as e:
        logger.error(f"Errore recupero log frontend: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Errore recupero log frontend", "log_text": ""},
        )


@app.get("/api/status")
def api_status():
    data, error = _backend_request("/status", timeout=10)

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": error or "Errore recupero status backend",
                "backend": "error",
                "database": "error",
                "ollama": "error",
                "esperto": "error",
            },
        )

    return JSONResponse(content=data)


@app.get("/api/domains")
def api_domains():
    data, error = _backend_request("/domains", timeout=30)

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": error or "Errore recupero domini",
                "domains": [],
            },
        )

    return JSONResponse(content=data)


@app.post("/api/parse")
def api_parse(request: ParseFrontendRequest):
    payload = {
        "url": request.url,
        "local": request.local,
    }

    if getattr(request, "html_text", None):
        payload["html_text"] = request.html_text

    data, error = _backend_request(
        "/parse",
        method="POST",
        json_data=payload,
        timeout=120,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore parsing"},
        )

    return JSONResponse(content=data)

@app.get("/api/gold_standard")
def api_gold_standard(url: str):
    data, error = _backend_request(
        "/gold_standard",
        params={"url": url},
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=404,
            content={"error": error or "Gold standard non trovato"},
        )

    return JSONResponse(content=data)


@app.get("/api/full_gold_standard")
def api_full_gold_standard(domain: str):
    data, error = _backend_request(
        "/full_gold_standard",
        params={"domain": domain},
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": error or "Gold standard non trovato",
                "gold_standard": [],
            },
        )

    return JSONResponse(content=data)


@app.get("/api/gold_standard_urls")
def api_gold_standard_urls(domain: str):
    data, error = _backend_request(
        "/gold_standard_urls",
        params={"domain": domain},
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": error or "Errore recupero URL gold standard",
                "gold_standard_urls": [],
            },
        )

    return JSONResponse(content=data)

@app.post("/api/evaluate_judge")
def api_evaluate_judge(request: EvaluateFrontendRequest):
    data, error = _backend_request(
        "/evaluate_judge",
        method="POST",
        json_data={
            "parsed_text": request.parsed_text,
            "gold_text": request.gold_text,
            "url": request.url,
        },
        timeout=300,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore evaluate_judge"},
        )

    return JSONResponse(content=data)


@app.get("/api/full_gs_eval")
def api_full_gs_eval(domain: str):
    data, error = _backend_request(
        "/full_gs_eval",
        params={"domain": domain},
        timeout=180,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore full_gs_eval"},
        )

    return JSONResponse(content=data)


@app.post("/api/add_web_resource")
def api_add_web_resource(request: AddWebResourceFrontendRequest):
    data, error = _backend_request(
        "/add_web_resource",
        method="POST",
        json_data={
            "url": request.url,
            "html_text": request.html_text,
            "title": request.title,
            "domain": request.domain,
        },
        timeout=120,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore add_web_resource"},
        )

    return JSONResponse(content=data)


@app.post("/api/add_gold_standard")
def api_add_gold_standard(request: AddGoldStandardFrontendRequest):
    data, error = _backend_request(
        "/add_gold_standard",
        method="POST",
        json_data={
            "url": request.url,
            "gold_text": request.gold_text,
        },
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore add_gold_standard"},
        )

    return JSONResponse(content=data)


@app.delete("/api/gold_standard")
def api_delete_gold_standard(request: UrlFrontendRequest):
    data, error = _backend_request(
        "/gold_standard",
        method="DELETE",
        json_data={
            "url": request.url,
        },
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore eliminazione gold standard"},
        )

    return JSONResponse(content=data)


@app.delete("/api/web_resource")
def api_delete_web_resource(request: UrlFrontendRequest):
    data, error = _backend_request(
        "/web_resource",
        method="DELETE",
        json_data={
            "url": request.url,
        },
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore eliminazione web resource"},
        )

    return JSONResponse(content=data)


@app.get("/api/db_stats")
def api_db_stats():
    data, error = _backend_request(
        "/db_stats",
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore recupero statistiche DB"},
        )

    return JSONResponse(content=data)


@app.get("/api/db_schema")
def api_db_schema():
    data, error = _backend_request(
        "/db_schema",
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore recupero schema DB"},
        )

    return JSONResponse(content=data)


@app.post("/api/evaluate")
def api_evaluate(request: EvaluateFrontendRequest):
    data, error = _backend_request(
        "/evaluate",
        method="POST",
        json_data={
            "parsed_text": request.parsed_text,
            "gold_text": request.gold_text,
            "url": request.url,
        },
        timeout=60,
        include_error_body=True,
    )

    if error or data is None:
        return JSONResponse(
            status_code=502,
            content={"error": error or "Errore evaluation"},
        )

    return JSONResponse(content=data)
