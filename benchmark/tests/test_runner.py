import json
from pathlib import Path

import pytest

from src import judge, ollama, runner
from src.runner import RunSpec, check_spec, execute_run

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
    def index_zip(self, content, filename):
        return {"indexing_seconds": 1.5}

    def health(self):
        return {"points_count": 42}

    def retrieve(self, question, use_planner, options):
        return {"subquestions": [question], "sources": SOURCES, "seconds": 0.1}

    def ask(self, question, use_planner, options):
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
    monkeypatch.setattr(judge, "judge_facts", lambda q, a, facts, **kw: [True] + [False] * (len(facts) - 1))
    monkeypatch.setattr(judge, "judge_faithful", lambda *a, **kw: True)
    monkeypatch.setattr(judge, "judge_abstains", lambda *a, **kw: False)
    monkeypatch.setattr(ollama, "chat", lambda *a, **kw: "risposta closed book")


def test_check_spec_rejects_unknown_configs():
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", retrieval_configs=["inventata"]))
    with pytest.raises(ValueError):
        check_spec(RunSpec(dataset="requests", retrieval_configs=["closed_book"]))
    check_spec(RunSpec(dataset="requests"))


def test_full_run_writes_all_artifacts(stubbed):
    spec = RunSpec(
        dataset="requests", limit=3, retrieval_configs=["full", "naive"],
        generation_configs=["full", "closed_book"], latency_questions=2, latency_repeats=1, latency_warmups=1,
    )
    execute_run("test-run", spec, client=FakeClient())

    run_dir = runner.RESULTS_DIR / "test-run"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "done", status

    retrieval = [json.loads(line) for line in (run_dir / "retrieval.jsonl").read_text().splitlines()]
    assert len(retrieval) == 6
    assert retrieval[0]["metrics"]["symbol_hit@1"] == 1.0  # lk01: default_user_agent e' la prima fonte

    generation = [json.loads(line) for line in (run_dir / "generation.jsonl").read_text().splitlines()]
    closed_book = [row for row in generation if row["config"] == "closed_book"]
    assert closed_book and all("faithfulness" not in row["metrics"] for row in closed_book)
    assert all(row["metrics"]["citation_in_context"] is None for row in closed_book)
    full = [row for row in generation if row["config"] == "full"]
    assert full[0]["metrics"]["citation_exists"] == 0.5  # utils.py esiste, fake.py no

    assert len((run_dir / "latency.jsonl").read_text().splitlines()) == 2  # 1 config x 2 domande x 1 ripetizione

    report = (run_dir / "report.md").read_text()
    for heading in ("## Retrieval", "## Generazione", "## Latenza", "Ablation"):
        assert heading in report
    assert (run_dir / "summary.csv").exists() and (run_dir / "manifest.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["dataset"]["commit"] == "abc1234567890"
    assert manifest["indexing_job"]["indexing_seconds"] == 1.5


def test_failed_run_records_error_in_status(monkeypatch):
    def boom(dataset):
        raise RuntimeError("git non raggiungibile")

    monkeypatch.setattr(runner, "fetch_repo", boom)
    execute_run("failing-run", RunSpec(dataset="requests"), client=FakeClient())
    status = json.loads((runner.RESULTS_DIR / "failing-run" / "status.json").read_text())
    assert status["status"] == "error" and "git non raggiungibile" in status["error"]
