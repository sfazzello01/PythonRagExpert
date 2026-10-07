import json
from pathlib import Path

import pytest

from src import ollama, report, runner
from src.dataset import load_dataset
from src.runner import RunSpec, check_spec, execute_run, judging_key, sample_questions

SOURCES = [
    {
        "filename": "src/requests/utils.py", "symbol_name": "default_user_agent", "chunk_type": "function",
        "start_line": 1, "end_line": 5, "chunk_id": "c1", "score": 0.9, "graph_relation": None,
        "content": "def default_user_agent(): ...",
    },
    {
        "filename": "src/requests/api.py", "symbol_name": "get", "chunk_type": "function",
        "start_line": 1, "end_line": 5, "chunk_id": "c2", "score": 0.5, "graph_relation": None,
        "content": "def get(): ...",
    },
]


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
        return {"subquestions": subquestions or [question, "sotto-domanda"], "sources": SOURCES, "seconds": 0.1}

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
    check_spec(RunSpec(dataset="requests"))


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
    assert sum("metrics" in row for row in retrieval) == 8  # panoramica e senza risposta non hanno gold

    generation = _rows("test-run", "generation.jsonl")
    closed_book = [row for row in generation if row["config"] == "closed_book"]
    assert closed_book and all(row["metrics"]["citation_in_context"] is None for row in closed_book)
    assert all("total_s" not in row["metrics"] for row in closed_book)
    full = [row for row in generation if row["config"] == "full"]
    assert full[0]["metrics"]["citation_exists"] == 0.5  # utils.py esiste, fake.py no
    assert all(row["metrics"]["total_s"] == 0.5 for row in full)  # tempi presi dalla stessa richiesta

    report = (run_dir / "report.md").read_text()
    for heading in ("## Retrieval", "## Generazione", "## Latenza", "Ablation", "Campione: 6 domande"):
        assert heading in report
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
    assert "0 risposte giudicate" in (run_dir / "report.md").read_text()

    judgments = [
        {"key": item["key"], "fact_verdicts": [True] * len(item["key_facts"]), "abstains": not item["key_facts"], "faithful": True}
        for item in items
        if keys[item["key"]]["config"] == "full"
    ]
    judgments[0]["fact_verdicts"] = [True] * 9  # numero di verdetti sbagliato: scartato
    (run_dir / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    report.write_report(run_dir, run_dir / "human_labels.json")

    text = (run_dir / "report.md").read_text()
    assert "5 risposte giudicate" in text
    full_row = next(line for line in text.splitlines() if line.startswith("| full | 6 |"))
    assert "1.00 [1.00, 1.00]" in full_row  # correttezza delle risposte giudicate


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
    assert "## Latenza" not in (runner.RESULTS_DIR / "failing-answers" / "report.md").read_text()
