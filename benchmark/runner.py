#!/usr/bin/env python3
"""Benchmark esterno del RAG per la codebase parser."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib import error, request

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ROUTES_RE = re.compile(r"@(?:app|router)\.(get|post|put|delete|patch)\(\s*['\"]([^'\"]+)")
FETCH_RE = re.compile(r"fetch\(\s*['\"`]([^'\"`]+)")
TIMESTAMP = time.strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configurazione validata del benchmark."""

    root: Path
    url: str
    target_paths: tuple[str, ...]
    extensions: frozenset[str]
    skip_dirs: frozenset[str]
    use_planner: bool
    use_reasoning: bool
    timeout: int
    chunk_size: int
    chunk_overlap: int
    output_limit: int
    error_limit: int
    static_only: bool


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
        root=Path(raw.get("root", REPO_ROOT)).resolve(),
        url=url.rstrip("/"),
        target_paths=tuple(raw["target_paths"]),
        extensions=frozenset(raw["supported_extensions"]),
        skip_dirs=frozenset(raw.get("skip_dirs", ())),
        use_planner=bool(raw.get("use_planner", True)),
        use_reasoning=bool(raw.get("use_reasoning", False)),
        timeout=timeout,
        chunk_size=int(raw.get("chunk_size", 1200)),
        chunk_overlap=int(raw.get("chunk_overlap", 150)),
        output_limit=int(raw.get("max_output_chars", 20000)),
        error_limit=int(raw.get("max_error_chars", 1000)),
        static_only=bool(args.static_only),
    )


def load_queries(path: Path) -> list[dict[str, str]]:
    """Carica esattamente 50 domande con ID e area unici/validi."""
    queries = load_json(path)
    if not isinstance(queries, list) or len(queries) != 50:
        raise ValueError("queries.json deve contenere esattamente 50 domande")
    ids: set[str] = set()
    for item in queries:
        query_id = str(item.get("id", ""))
        if not query_id or query_id in ids:
            raise ValueError(f"ID domanda mancante o duplicato: {query_id!r}")
        if not str(item.get("area", "")).strip() or not str(item.get("question", "")).strip():
            raise ValueError(f"Domanda non valida per ID {query_id!r}")
        ids.add(query_id)
    return queries


def iter_files(config: BenchmarkConfig) -> Iterator[Path]:
    """Enumera i file supportati in una sola visita deterministica."""
    for target in config.target_paths:
        base = config.root / target
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in candidates:
            if path.is_file() and path.suffix in config.extensions:
                if not any(part in config.skip_dirs for part in path.relative_to(config.root).parts):
                    yield path


def text_chunks(size: int, chunk_size: int, overlap: int) -> int:
    """Stima il numero di chunk di fallback con la regola documentata."""
    if size <= 0:
        return 0
    step = max(1, chunk_size - overlap)
    return 1 + max(0, (size - chunk_size + step - 1) // step)


def inspect_python(path: Path, text: str, chunk_size: int, overlap: int) -> dict[str, Any]:
    """Estrae in una sola passata simboli, chiamate, ereditarietà e route."""
    data: dict[str, Any] = {
        "routes": ROUTES_RE.findall(text),
        "classes": 0,
        "functions": 0,
        "nodes": 0,
        "edges": 0,
        "chunks": 0,
        "parse_error": None,
    }
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        data["parse_error"] = f"{exc.__class__.__name__}: {exc}"
        data["chunks"] = text_chunks(len(text), chunk_size, overlap)
        return data

    names: set[str] = set()
    calls: set[tuple[str, str]] = set()
    parents: set[tuple[str, str]] = set()

    def full_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{full_name(node.value)}.{node.attr}"
        return ""

    class Visitor(ast.NodeVisitor):
        """Visitor con scope stack per attribuire le chiamate al simbolo corrente."""

        def __init__(self) -> None:
            self.scopes: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            data["classes"] += 1
            names.add(node.name)
            for base in node.bases:
                parent = full_name(base)
                if parent:
                    parents.add((node.name, parent))
            self.scopes.append(node.name)
            self.generic_visit(node)
            self.scopes.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            data["functions"] += 1
            names.add(node.name)
            self.scopes.append(node.name)
            self.generic_visit(node)
            self.scopes.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

        def visit_Call(self, node: ast.Call) -> None:
            callee = full_name(node.func)
            if callee:
                calls.add((self.scopes[-1] if self.scopes else "module", callee))
            self.generic_visit(node)

    Visitor().visit(tree)
    data["nodes"] = len(names)
    data["edges"] = len(calls) + len(parents)
    data["chunks"] = data["classes"] + data["functions"]
    return data


def static_metrics(config: BenchmarkConfig) -> dict[str, Any]:
    """Calcola metriche locali aggregate durante un'unica scansione."""
    files = list(iter_files(config))
    services = Counter()
    extensions = Counter()
    routes: list[dict[str, str]] = []
    fetches: list[dict[str, str]] = []
    totals = Counter()
    errors: list[dict[str, str]] = []

    for path in files:
        relative = path.relative_to(config.root).as_posix()
        service = relative.split("/", 1)[0]
        services[service] += 1
        extensions[path.suffix] += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append({"file": relative, "error": str(exc)[: config.error_limit]})
            continue

        totals["files"] += 1
        totals["bytes"] += len(text.encode("utf-8"))
        if path.suffix == ".py":
            totals["python_files"] += 1
            info = inspect_python(path, text, config.chunk_size, config.chunk_overlap)
            totals.update(classes=info["classes"], functions=info["functions"])
            totals.update(ast_nodes=info["nodes"], ast_edges=info["edges"], chunks=info["chunks"])
            routes.extend(
                {"file": relative, "method": method.upper(), "path": endpoint}
                for method, endpoint in info["routes"]
            )
            if info["parse_error"]:
                errors.append({"file": relative, "error": info["parse_error"]})
        else:
            totals["chunks"] += text_chunks(len(text), config.chunk_size, config.chunk_overlap)

        fetches.extend(
            {"file": relative, "url": match}
            for match in FETCH_RE.findall(text)
            if path.suffix in {".html", ".js"}
        )

    return {
        "kind": "independent_estimate",
        "files": totals["files"],
        "python_files": totals["python_files"],
        "bytes": totals["bytes"],
        "estimated_chunks": totals["chunks"],
        "python_classes": totals["classes"],
        "python_functions": totals["functions"],
        "ast_nodes": totals["ast_nodes"],
        "ast_edges_estimate": totals["ast_edges"],
        "fastapi_routes": routes,
        "frontend_fetches": fetches,
        "services": dict(sorted(services.items())),
        "extensions": dict(sorted(extensions.items())),
        "read_or_parse_errors": errors,
    }


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


def ask_expert(query: dict[str, str], config: BenchmarkConfig) -> dict[str, Any]:
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


def run(config: BenchmarkConfig, queries: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Esegue una run completa: metriche statiche, health e 50 richieste."""
    started = time.perf_counter()
    report: dict[str, Any] = {
        "benchmark_timestamp": TIMESTAMP,
        "config": {
            "esperto_url": config.url,
            "use_planner": config.use_planner,
            "use_reasoning": config.use_reasoning,
            "timeout_seconds": config.timeout,
        },
        "static": static_metrics(config),
    }
    answers: list[dict[str, Any]] = []
    _, answers_path = output_paths()
    if config.static_only:
        report["runtime"] = {"skipped": True, "reason": "static_only"}
        return report, answers

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
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report, answers


def parse_args() -> argparse.Namespace:
    """Definisce le opzioni CLI del benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark esterno RAG della codebase parser")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="file di configurazione JSON")
    parser.add_argument("--queries", default=str(ROOT / "queries.json"), help="file con 50 domande")
    parser.add_argument("--esperto-url", help="override URL esperto")
    parser.add_argument("--static-only", action="store_true", help="salta le chiamate all'API")
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
