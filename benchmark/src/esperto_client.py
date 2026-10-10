"""Client HTTP verso i servizi di esperto (indexer e query)."""

from __future__ import annotations

import json
import time

import httpx

from config import ESPERTO_URL, INDEX_TIMEOUT_SECONDS, INDEXER_URL, REQUEST_TIMEOUT


class EspertoClient:
    """Chiama le API di esperto: stato, indicizzazione, retrieval e risposta."""

    def __init__(self, http: httpx.Client | None = None):
        """Crea il client.

        Args:
            http: Client HTTP da usare; se manca ne viene creato uno con il timeout configurato.
        """
        self._http = http or httpx.Client(timeout=REQUEST_TIMEOUT)

    def health(self) -> dict:
        """Interroga lo stato di esperto.

        Returns:
            La risposta di `/health`: stato dell'indice e numero di punti.
        """
        response = self._http.get(f"{ESPERTO_URL}/health")
        response.raise_for_status()
        return response.json()

    def indexed_filename(self) -> str | None:
        """Legge il nome dello ZIP da cui è stato costruito l'indice attivo.

        Returns:
            Il nome del file, oppure None se l'ultimo job non è andato a buon fine.
        """
        status = self._http.get(f"{INDEXER_URL}/status").json()
        return status["job"]["filename"] if status["status"] == "ready" else None

    def index_zip(self, content: bytes, filename: str, poll_seconds: float = 5.0) -> dict:
        """Carica lo ZIP e attende la fine del job di indicizzazione.

        Args:
            content: Byte dell'archivio.
            filename: Nome dell'archivio.
            poll_seconds: Intervallo tra due controlli dello stato del job.

        Returns:
            Il job completato, con stato e durata dell'indicizzazione.

        Raises:
            RuntimeError: Se l'indicizzazione fallisce.
            TimeoutError: Se l'indicizzazione non termina entro il tempo massimo.
        """
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

    def retrieve(
        self, question: str, use_planner: bool, options: dict, subquestions: list[str] | None = None
    ) -> dict:
        """Esegue planner e retrieval, oppure il solo retrieval su un piano già calcolato.

        Args:
            question: Domanda da porre.
            use_planner: True per scomporre la domanda con il planner.
            options: Opzioni di retrieval da disattivare o fissare.
            subquestions: Piano già calcolato; se presente il planner non viene chiamato.

        Returns:
            Sotto-domande, fonti e candidati esclusi, più la durata della richiesta in `seconds`.
        """
        start = time.perf_counter()
        response = self._http.post(
            f"{ESPERTO_URL}/retrieve",
            json={"message": question, "use_planner": use_planner, "retrieval": options, "subquestions": subquestions},
        )
        response.raise_for_status()
        return {**response.json(), "seconds": time.perf_counter() - start}

    def ask(self, question: str, use_planner: bool, options: dict) -> dict:
        """Pone la domanda e consuma lo stream NDJSON di `/ask`, misurandone i tempi.

        Args:
            question: Domanda da porre.
            use_planner: True per scomporre la domanda con il planner.
            options: Opzioni di retrieval da disattivare o fissare.

        Returns:
            Risposta, fonti, sotto-domande e mappa, con i tempi di arrivo di fonti, primo token e fine
            risposta; `error` contiene l'eventuale errore emesso a metà stream.
        """
        result = {
            "subquestions": [], "sources": [], "repo_map": "", "answer": "",
            "retrieval_s": None, "ttft_s": None, "error": None,
        }
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
                    result["error"] = line[:200]  # es. "[ERROR] ..." emesso da esperto a metà stream
                    continue
                now = time.perf_counter() - start
                if event["type"] == "sources":
                    result["sources"] = event["data"]["sources"]
                    result["subquestions"] = event["data"]["subquestions"]
                    result["repo_map"] = event["data"].get("repo_map", "")
                    result["retrieval_s"] = now
                elif event["type"] == "answer":
                    if result["ttft_s"] is None:
                        result["ttft_s"] = now
                    result["answer"] += event["data"]
        result["total_s"] = time.perf_counter() - start
        return result
