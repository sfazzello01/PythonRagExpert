import json
from pathlib import Path

import pytest

from src import ollama, report, runner
from src.dataset import load_dataset
from src.runner import RunSpec, check_spec, execute_run, judging_key, sample_questions

SOURCES = [
    {
        "filename": "src/requests/utils.py", "symbol_name": "default_user_agent", "chunk_type": "function",
        "start_line": 1, "end_line": 5, "chunk_id": "c1", "score": 0.9, "origin": "search", "graph_relation": None,
        "content": "def default_user_agent(): ...",
    },
    {
        "filename": "src/requests/api.py", "symbol_name": "get", "chunk_type": "function",
        "start_line": 1, "end_line": 5, "chunk_id": "c2", "score": 0.5, "origin": "graph", "graph_relation": "calls",
        "content": "def get(): ...",
    },
]
DROPPED = [{"filename": "src/requests/sessions.py", "symbol_name": "Session.send", "chunk_id": "c3", "origin": "search"}]


class FakeClient:
    def __init__(self):
        self.plans = []  # piano ricevuto da ogni chiamata a retrieve
        self.asks = 0

    def index_zip(self, content, filename):
        return {"indexing_seconds": 1.5}

    def indexed_filename(self):
        return "requests.zip"

    def health(self):
        return {"status": "ok", "points_count": 42}

    def retrieve(self, question, use_planner, options, subquestions=None):
        self.plans.append(subquestions)
        return {
            "subquestions": subquestions or [question, "sotto-domanda"], "sources": SOURCES, "dropped": DROPPED,
            "seconds": 0.1,
        }

    def ask(self, question, use_planner, options):
        self.asks += 1
        return {
            "subquestions": [question], "sources": SOURCES, "answer": "Vedi src/requests/utils.py e fake.py",
            "retrieval_s": 0.1, "ttft_s": 0.2, "total_s": 0.5, "error": None,
        }


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    monkeypatch.setattr(runner, "fetch_repo", lambda dataset: (tmp_path, "abc1234567890"))
    monkeypatch.setattr(runner, "validate_dataset", lambda dataset, repo_dir: [])
    monkeypatch.setattr(runner, "list_repo_files", lambda repo_dir: ["src/requests/utils.py"])
    monkeypatch.setattr(runner, "build_zip", lambda repo_dir, files: b"zip")
    monkeypatch.setattr(ollama, "chat", lambda *a, **kw: "risposta closed book")


def _rows(run_id, name):
    path = runner.RESULTS_DIR / run_id / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_check_spec_rejects_unknown_configs():
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", retrieval_configs=["inventata"]))
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", retrieval_configs=["closed_book"]))
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", stages=["latency"]))
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", retrieval_configs=["full@k=molti"]))
    check_spec(RunSpec(dataset="requests"))


def test_any_config_can_fix_its_k():
    assert runner.resolve_config("full@k=20") == (True, {"k": 20})
    assert runner.resolve_config("naive@k=50") == runner.resolve_config("naive_budget")
    check_spec(RunSpec(dataset="requests", retrieval_configs=["no_symbol_graph@k=8"], generation_configs=["full@k=20"]))


def test_sample_is_stratified_and_reproducible():
    questions = load_dataset("requests").questions
    sample = sample_questions(questions, 2, seed=0)
    assert sorted(q.category for q in sample) == sorted(
        ["lookup", "behavior", "multihop", "architecture", "overview", "unanswerable"] * 2
    )
    assert [q.id for q in sample] == [q.id for q in sample_questions(questions, 2, seed=0)]
    assert [q.id for q in sample] != [q.id for q in sample_questions(questions, 2, seed=1)]
    assert sample_questions(questions, None, seed=0) == questions


def test_full_run_writes_all_artifacts(stubbed):
    spec = RunSpec(
        dataset="requests", per_category=1, retrieval_configs=["full", "naive"],
        generation_configs=["full", "closed_book"],
    )
    execute_run("test-run", spec, client=FakeClient())

    run_dir = runner.RESULTS_DIR / "test-run"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "done", status

    retrieval = _rows("test-run", "retrieval.jsonl")
    assert len(retrieval) == 12  # 2 configurazioni x 6 domande (una per categoria)
    with_gold = [row for row in retrieval if "metrics" in row]
    assert len(with_gold) == 8  # panoramica e senza risposta non hanno gold
    # Per ogni simbolo atteso la riga dice dove e' finito: e' la base della diagnosi.
    statuses = {item["status"] for row in with_gold for item in row["gold"]}
    assert statuses <= {"search", "symbol", "same_file", "graph", "budget", "missing"} and "missing" in statuses
    assert all(row["dropped"] == DROPPED and row["sources"][1]["origin"] == "graph" for row in retrieval)

    generation = _rows("test-run", "generation.jsonl")
    closed_book = [row for row in generation if row["config"] == "closed_book"]
    assert closed_book and all("total_s" not in row["metrics"] for row in closed_book)
    assert all("symbol_context_recall" not in row["metrics"] for row in closed_book)
    full = [row for row in generation if row["config"] == "full"]
    assert all(row["metrics"]["total_s"] == 0.5 for row in full)  # tempi presi dalla stessa richiesta
    assert all(row["metrics"]["answer_chars"] == len("Vedi src/requests/utils.py e fake.py") for row in full)
    # Il recall del contesto serve a separare gli errori del retrieval da quelli del modello.
    assert sum("symbol_context_recall" in row["metrics"] for row in full) == 4

    report = (run_dir / "report.md").read_text()
    for heading in (
        "## Diagnosi", "## Retrieval", "### Dove finiscono i simboli attesi", "### Domande con simboli mancanti",
        "## Generazione", "### Latenza", "Campione: 6 domande", "planner acceso",
    ):
        assert heading in report, heading
    assert "**Simboli attesi** (`full`)" in report and "rispetto a `naive`" in report
    assert (run_dir / "summary.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["dataset"]["commit"] == "abc1234567890"
    assert manifest["dataset"]["sampled_questions"] == 6
    assert manifest["indexing_job"]["indexing_seconds"] == 1.5


def test_planner_plan_is_computed_once_and_shared_by_ablations(stubbed):
    client = FakeClient()
    spec = RunSpec(
        dataset="requests", per_category=1, stages=["retrieval"],
        retrieval_configs=["full", "no_symbol_graph", "no_repo_map", "no_planner"],
    )
    execute_run("plan-run", spec, client=client)

    full, shared, own_plan, unplanned = (client.plans[i : i + 6] for i in range(0, 24, 6))
    assert all(plan is None for plan in full + own_plan + unplanned)  # senza mappa il piano cambia
    assert all(plan and plan[-1] == "sotto-domanda" for plan in shared)


def test_resume_skips_completed_questions(stubbed):
    spec = RunSpec(dataset="requests", per_category=1, retrieval_configs=["full"], generation_configs=["full"])
    execute_run("resumed-run", spec, client=FakeClient())
    client = FakeClient()
    execute_run("resumed-run", spec.model_copy(update={"skip_index": True}), client=client)

    assert client.plans == [] and client.asks == 1  # solo il warmup
    assert len(_rows("resumed-run", "retrieval.jsonl")) == 6
    assert len(_rows("resumed-run", "generation.jsonl")) == 6


def test_skip_index_requires_the_dataset_index(stubbed):
    class OtherIndex(FakeClient):
        def indexed_filename(self):
            return "altro.zip"

    execute_run("wrong-index", RunSpec(dataset="requests", skip_index=True), client=OtherIndex())
    status = json.loads((runner.RESULTS_DIR / "wrong-index" / "status.json").read_text())
    assert status["status"] == "error" and "skip_index" in status["error"]


def test_failed_run_records_error_in_status(monkeypatch):
    def boom(dataset):
        raise RuntimeError("git non raggiungibile")

    monkeypatch.setattr(runner, "fetch_repo", boom)
    execute_run("failing-run", RunSpec(dataset="requests"), client=FakeClient())
    status = json.loads((runner.RESULTS_DIR / "failing-run" / "status.json").read_text())
    assert status["status"] == "error" and "git non raggiungibile" in status["error"]


def test_answers_are_exported_blind_and_external_judgments_feed_the_report(stubbed):
    spec = RunSpec(dataset="requests", per_category=1, stages=["generation"], generation_configs=["full", "closed_book"])
    execute_run("judged-run", spec, client=FakeClient())
    run_dir = runner.RESULTS_DIR / "judged-run"

    items = _rows("judged-run", "to_judge.jsonl")
    assert len(items) == 12 and all(set(item) == {"key", "question", "key_facts", "answer"} for item in items)
    assert [item["key"] for item in items] == sorted(item["key"] for item in items)  # ordine indipendente dalla config
    generation = _rows("judged-run", "generation.jsonl")
    keys = {judging_key("judged-run", row["config"], row["id"]): row for row in generation}
    assert set(keys) == {item["key"] for item in items}
    # Il contesto e' salvato solo per le risposte che ne hanno uno.
    assert {path.stem for path in (run_dir / "contexts").iterdir()} == {k for k, row in keys.items() if row["config"] == "full"}
    assert "Risposte giudicate: 0" in (run_dir / "report.md").read_text()

    judgments = [
        {"key": item["key"], "fact_verdicts": [True] * len(item["key_facts"]), "abstains": not item["key_facts"]}
        for item in items
        if keys[item["key"]]["config"] == "full"
    ]
    judgments[0]["fact_verdicts"] = [True] * 9  # numero di verdetti sbagliato: scartato
    wrong = next(j for j in judgments[1:] if j["fact_verdicts"])
    wrong["fact_verdicts"][0] = False
    wrong_fact = next(item["key_facts"][0] for item in items if item["key"] == wrong["key"])
    (run_dir / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    report.write_report(run_dir)

    text = (run_dir / "report.md").read_text()
    assert "Risposte giudicate: 5" in text
    # Il fatto sbagliato e' elencato, e la correttezza e' separata per completezza del contesto.
    assert "### Fatti sbagliati" in text and wrong_fact in text.split("### Fatti sbagliati")[1]
    assert "### Retrieval o modello?" in text and "**Retrieval o modello?** (`full`)" in text


def test_two_runs_are_compared_on_their_common_questions(stubbed):
    spec = RunSpec(dataset="requests", per_category=1, stages=["retrieval"], retrieval_configs=["full", "naive"])
    execute_run("compare-a", spec, client=FakeClient())
    execute_run("compare-b", spec.model_copy(update={"retrieval_configs": ["full"]}), client=FakeClient())

    text = report.compare_runs(runner.RESULTS_DIR / "compare-a", runner.RESULTS_DIR / "compare-b")
    rows = [line for line in text.splitlines() if line.startswith("| symbol_")]
    assert all("| full |" in row and "+0.00 [+0.00, +0.00]" in row for row in rows) and len(rows) >= 2
    assert "naive" not in text  # solo le configurazioni presenti in entrambi i run


def test_failed_answers_are_not_exported_for_judging(stubbed):
    class FailingClient(FakeClient):
        def ask(self, question, use_planner, options):
            result = super().ask(question, use_planner, options)
            return result if self.asks == 1 else {**result, "answer": "", "error": "[ERROR] timeout"}  # warmup ok

    spec = RunSpec(dataset="requests", per_category=1, stages=["generation"], generation_configs=["full"])
    execute_run("failing-answers", spec, client=FailingClient())

    generation = _rows("failing-answers", "generation.jsonl")
    assert len(generation) == 6 and all(row["metrics"] == {"answer_error": 1.0} for row in generation)
    assert _rows("failing-answers", "to_judge.jsonl") == []
    assert "### Latenza" not in (runner.RESULTS_DIR / "failing-answers" / "report.md").read_text()
