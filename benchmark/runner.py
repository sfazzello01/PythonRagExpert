#!/usr/bin/env python3
"""Benchmark esterno del RAG per la codebase parser."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
TIMESTAMP = time.strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configurazione validata del benchmark."""

    url: str
    use_planner: bool
    use_reasoning: bool
    timeout: int
    output_limit: int
    error_limit: int


def load_json(path: Path) -> Any:
    """Legge un file JSON con errore contestualizzato."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON non valido in {path}: {exc}") from exc


def load_config(path: Path, args: argparse.Namespace) -> BenchmarkConfig:
    """Costruisce la configurazione finale applicando gli override CLI."""
    raw = load_json(path)
    env_url = os.getenv("ESPERTO_URL")
    url = args.esperto_url or env_url or raw.get("esperto_url", "http://localhost:5000")
    timeout = int(raw.get("request_timeout_seconds", 120))
    return BenchmarkConfig(
        url=url.rstrip("/"),
        use_planner=bool(raw.get("use_planner", True)),
        use_reasoning=bool(raw.get("use_reasoning", False)),
        timeout=timeout,
        output_limit=int(raw.get("max_output_chars", 20000)),
        error_limit=int(raw.get("max_error_chars", 1000)),
    )


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Carica esattamente 50 domande con ground truth retrieval valido."""
    queries = load_json(path)
    if not isinstance(queries, list) or len(queries) != 50:
        raise ValueError("queries.json deve contenere esattamente 50 domande")
    ids: set[str] = set()
    for item in queries:
        query_id = str(item.get("id", ""))
        if not query_id or query_id in ids:
            raise ValueError(f"ID domanda mancante o duplicato: {query_id!r}")
        expected = item.get("relevant_sources")
        if not str(item.get("area", "")).strip() or not str(item.get("question", "")).strip():
            raise ValueError(f"Domanda non valida per ID {query_id!r}")
        if not isinstance(expected, list) or not expected or not all(isinstance(source, str) and source.strip() for source in expected):
            raise ValueError(f"relevant_sources mancante o vuoto per ID {query_id!r}")
        ids.add(query_id)
    return queries


def trim(value: str, limit: int) -> str:
    """Limita una stringa preservando un suffisso di troncamento."""
    return value if len(value) <= limit else value[:limit] + "…[troncato]"


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: int) -> Any:
    """Esegue una richiesta HTTP JSON e restituisce la risposta decodificata."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_stream(response: Any, config: BenchmarkConfig) -> tuple[str, list[Any], list[str], list[str]]:
    """Consuma lo streaming JSON dell'esperto e separa risposta e metadati."""
    answer_parts: list[str] = []
    reasoning: list[str] = []
    sources: list[Any] = []
    errors: list[str] = []
    for raw_line in response:
        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type")
            data = event.get("data")
            if event_type == "answer":
                answer_parts.append(str(data))
            elif event_type == "reasoning" and config.use_reasoning:
                reasoning.append(str(data))
            elif event_type == "sources":
                sources = list(data.get("sources", [])) if isinstance(data, dict) else []
        except (json.JSONDecodeError, AttributeError) as exc:
            errors.append(trim(str(exc), config.error_limit))
    return "".join(answer_parts), sources, reasoning, errors


def normalize_source_path(value: str) -> str:
    """Rende confrontabili i path locali e quelli storici dell'indice."""
    path = value.replace("\\", "/").split("#", 1)[0].strip().strip("/ ")
    path = path.removeprefix("labnew/")
    for prefix in ("codebase/", "repository/"):
        path = path.removeprefix(prefix)
    for marker in ("backend/", "frontend/"):
        index = path.find(marker)
        if index >= 0:
            return path[index:]
    return path


def source_filename(source: Any) -> str:
    """Estrae il filename dai diversi formati di source dell'API."""
    if isinstance(source, dict):
        return str(source.get("filename") or source.get("file") or "")
    return str(source)


def source_matches(expected: str, returned: Any) -> bool:
    """Confronta una fonte attesa con una fonte restituita dall'esperto."""
    expected_path = normalize_source_path(expected)
    returned_path = normalize_source_path(source_filename(returned))
    return bool(expected_path and returned_path and (expected_path == returned_path or returned_path.endswith("/" + expected_path)))


def retrieval_metrics(expected: list[str], returned: list[Any], cutoff: int = 5) -> dict[str, float | int | None]:
    """Calcola metriche retrieval sulle prime fonti restituite, senza duplicati."""
    unique: list[Any] = []
    seen: set[str] = set()
    for source in returned:
        filename = normalize_source_path(source_filename(source))
        if filename and filename not in seen:
            seen.add(filename)
            unique.append(source)
    top_sources = unique[:cutoff]
    relevant = {normalize_source_path(source) for source in expected if normalize_source_path(source)}
    matches = [source for source in top_sources if any(source_matches(item, source) for item in relevant)]
    true_positives = len(matches)
    precision = true_positives / len(top_sources) if top_sources else 0.0
    recall = true_positives / len(relevant) if relevant else None
    first_rank = next((index for index, source in enumerate(top_sources, start=1) if any(source_matches(item, source) for item in relevant)), None)
    return {
        "precision_at_5": round(precision, 4),
        "recall_at_5": round(recall, 4) if recall is not None else None,
        "hit_at_5": int(first_rank is not None),
        "mrr": round(1 / first_rank, 4) if first_rank else 0.0,
        "expected_sources": len(relevant),
        "unique_returned_sources": len(unique),
    }


def aggregate_retrieval(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggrega metriche retrieval globali e per area."""
    metric_names = ("precision_at_5", "recall_at_5", "hit_at_5", "mrr")
    valid = [item["retrieval_metrics"] for item in answers if item.get("retrieval_metrics")]

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"questions": len(items)}
        for name in metric_names:
            values = [item[name] for item in items if item.get(name) is not None]
            result[f"avg_{name}"] = round(sum(values) / len(values), 4) if values else None
            result[f"median_{name}"] = round(float(median(values)), 4) if values else None
        return result

    by_area: dict[str, list[dict[str, Any]]] = {}
    for answer in answers:
        metrics = answer.get("retrieval_metrics")
        if metrics:
            by_area.setdefault(answer["area"], []).append(metrics)
    return {**summarize(valid), "by_area": {area: summarize(items) for area, items in sorted(by_area.items())}}


def ask_expert(query: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    """Interroga l'esperto una sola volta e raccoglie misura e risposta."""
    started = time.perf_counter()
    payload = {
        "message": query["question"],
        "use_planner": config.use_planner,
        "use_reasoning": config.use_reasoning,
    }
    result: dict[str, Any] = {
        "id": query["id"],
        "area": query["area"],
        "question": query["question"],
        "answer": "",
        "sources": [],
        "sources_count": 0,
        "relevant_sources": query["relevant_sources"],
        "retrieval_metrics": None,
        "latency_seconds": None,
        "status": "failed",
        "error": None,
        "timestamp": TIMESTAMP,
    }
    try:
        req = request.Request(
            f"{config.url}/ask",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=config.timeout) as response:
            answer, sources, reasoning, stream_errors = parse_stream(response, config)
        result.update(
            answer=trim(answer, config.output_limit),
            sources=sources,
            sources_count=len(sources),
            retrieval_metrics=retrieval_metrics(query["relevant_sources"], sources),
            latency_seconds=round(time.perf_counter() - started, 3),
            status="ok" if answer and not stream_errors else "partial",
            error="; ".join(stream_errors) or None,
        )
        if config.use_reasoning and reasoning:
            result["reasoning_chars"] = sum(len(part) for part in reasoning)
    except (OSError, TimeoutError, error.URLError, error.HTTPError) as exc:
        result.update(
            latency_seconds=round(time.perf_counter() - started, 3),
            error=trim(f"{exc.__class__.__name__}: {exc}", config.error_limit),
        )
    return result


def runtime_metrics(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Riassume le metriche RAG osservate dalle 50 richieste."""
    latencies = [item["latency_seconds"] for item in answers if item["latency_seconds"] is not None]
    sources = [item["sources_count"] for item in answers]
    completed = sum(item["status"] == "ok" for item in answers)
    partial = sum(item["status"] == "partial" for item in answers)
    failed = sum(item["status"] == "failed" for item in answers)
    return {
        "questions": len(answers),
        "completed": completed,
        "partial": partial,
        "failed": failed,
        "completion_rate": completed / len(answers) if answers else 0.0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "avg_sources": round(sum(sources) / len(sources), 3) if sources else None,
        "answers_with_sources": sum(bool(count) for count in sources),
        "by_area": dict(sorted(Counter(item["area"] for item in answers).items())),
    }


def output_paths() -> tuple[Path, Path]:
    """Prepara le directory e restituisce i percorsi degli output della run."""
    results_dir = ROOT / "results"
    answers_dir = ROOT / "answers"
    results_dir.mkdir(exist_ok=True)
    answers_dir.mkdir(exist_ok=True)
    report_path = results_dir / f"{TIMESTAMP}.json"
    answers_path = answers_dir / f"{TIMESTAMP}.jsonl"
    answers_path.touch()
    return report_path, answers_path


def write_outputs(report_path: Path, answers_path: Path, report: dict[str, Any]) -> None:
    """Scrive il report tecnico finale della run."""
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run(config: BenchmarkConfig, queries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Esegue una run completa via API: health e 50 richieste."""
    started = time.perf_counter()
    report: dict[str, Any] = {
        "benchmark_timestamp": TIMESTAMP,
        "config": {
            "esperto_url": config.url,
            "use_planner": config.use_planner,
            "use_reasoning": config.use_reasoning,
            "timeout_seconds": config.timeout,
        },
    }
    answers: list[dict[str, Any]] = []
    _, answers_path = output_paths()

    try:
        report["health"] = http_json("GET", f"{config.url}/health", None, config.timeout)
    except (OSError, TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
        report["health"] = {"status": "unavailable", "error": trim(str(exc), config.error_limit)}

    with answers_path.open("w", encoding="utf-8") as answers_file:
        for index, query in enumerate(queries, start=1):
            print(f"[{index:02d}/{len(queries)}] {query['id']}", flush=True)
            answer = ask_expert(query, config)
            answers.append(answer)
            answers_file.write(json.dumps(answer, ensure_ascii=False) + "\n")
            answers_file.flush()

    report["runtime"] = runtime_metrics(answers)
    report["retrieval_metrics"] = aggregate_retrieval(answers)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report, answers


def parse_args() -> argparse.Namespace:
    """Definisce le opzioni CLI del benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark esterno RAG della codebase parser")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="file di configurazione JSON")
    parser.add_argument("--queries", default=str(ROOT / "queries.json"), help="file con 50 domande")
    parser.add_argument("--esperto-url", help="override URL esperto")
    return parser.parse_args()


def main() -> int:
    """Punto di ingresso CLI."""
    try:
        args = parse_args()
        config = load_config(Path(args.config), args)
        queries = load_queries(Path(args.queries))
        report, _ = run(config, queries)
        report_path, answers_path = output_paths()
        write_outputs(report_path, answers_path, report)
    except (OSError, ValueError) as exc:
        print(f"benchmark failed: {exc}")
        return 1
    print(f"report: {report_path.relative_to(REPO_ROOT)}")
    print(f"answers: {answers_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
