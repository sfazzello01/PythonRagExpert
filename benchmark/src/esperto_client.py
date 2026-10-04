"""Client HTTP verso i servizi di esperto (indexer e query)."""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from config import ESPERTO_URL, INDEX_TIMEOUT_SECONDS, INDEXER_URL, REQUEST_TIMEOUT


class EspertoClient:
    def __init__(self, http: Optional[httpx.Client] = None):
        self._http = http or httpx.Client(timeout=REQUEST_TIMEOUT)

    def health(self) -> dict:
        response = self._http.get(f"{ESPERTO_URL}/health")
        response.raise_for_status()
        return response.json()

    def index_zip(self, content: bytes, filename: str, poll_seconds: float = 5.0) -> dict:
        """Carica lo ZIP e attende la fine del job; restituisce il job (status, indexing_seconds, ...)."""
        response = self._http.post(f"{INDEXER_URL}/upload", files={"file": (filename, content, "application/zip")})
        response.raise_for_status()
        deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            status = self._http.get(f"{INDEXER_URL}/status").json()
            if status["status"] == "ready":
                return status["job"]
            if status["status"] == "error":
                raise RuntimeError(f"Indicizzazione fallita: {status['job'].get('error')}")
            time.sleep(poll_seconds)
        raise TimeoutError("Timeout in attesa dell'indicizzazione")

    def retrieve(self, question: str, use_planner: bool, options: dict) -> dict:
        """Solo planner + retrieval. Aggiunge `seconds` (latenza end-to-end della chiamata)."""
        start = time.perf_counter()
        response = self._http.post(
            f"{ESPERTO_URL}/retrieve",
            json={"message": question, "use_planner": use_planner, "retrieval": options},
        )
        response.raise_for_status()
        return {**response.json(), "seconds": time.perf_counter() - start}

    def ask(self, question: str, use_planner: bool, options: dict) -> dict:
        """Consuma lo stream NDJSON di /ask misurando tempi di fonti, primo token e totale."""
        result = {"subquestions": [], "sources": [], "answer": "", "retrieval_s": None, "ttft_s": None, "error": None}
        start = time.perf_counter()
        payload = {"message": question, "use_planner": use_planner, "retrieval": options}
        with self._http.stream("POST", f"{ESPERTO_URL}/ask", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    result["error"] = line[:200]  # es. "[ERROR] ..." emesso da esperto a meta' stream
                    continue
                now = time.perf_counter() - start
                if event["type"] == "sources":
                    result["sources"] = event["data"]["sources"]
                    result["subquestions"] = event["data"]["subquestions"]
                    result["retrieval_s"] = now
                elif event["type"] == "answer":
                    if result["ttft_s"] is None:
                        result["ttft_s"] = now
                    result["answer"] += event["data"]
        result["total_s"] = time.perf_counter() - start
        return result
