"""Test isolati del benchmark esterno."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmark import runner


class RunnerTest(unittest.TestCase):
    def test_queries_schema_and_count(self) -> None:
        queries = runner.load_queries(runner.ROOT / "queries.json")
        self.assertEqual(50, len(queries))
        self.assertEqual(50, len({item["id"] for item in queries}))
        self.assertTrue(all(item["area"] and item["question"] for item in queries))

    def test_text_chunks(self) -> None:
        self.assertEqual(0, runner.text_chunks(0, 1200, 150))
        self.assertEqual(1, runner.text_chunks(100, 1200, 150))
        self.assertEqual(1, runner.text_chunks(1200, 1200, 150))
        self.assertEqual(2, runner.text_chunks(1201, 1200, 150))
        self.assertEqual(2, runner.text_chunks(2250, 1200, 150))
        self.assertEqual(3, runner.text_chunks(2251, 1200, 150))

    def test_inspect_python_single_pass(self) -> None:
        source = """from fastapi import FastAPI\napp = FastAPI()\nclass Base: pass\nclass Child(Base):\n    def method(self): helper()\ndef helper(): return 1\n@app.get('/x')\ndef endpoint(): helper()\n"""
        info = runner.inspect_python(Path("sample.py"), source, 1200, 150)
        self.assertEqual(2, info["classes"])
        self.assertEqual(3, info["functions"])
        self.assertEqual(5, info["nodes"])
        self.assertIn(("get", "/x"), info["routes"])
        self.assertEqual(5, info["edges"])

    def test_parse_stream(self) -> None:
        payload = [
            json.dumps({"type": "sources", "data": {"sources": [{"filename": "a.py"}]}}),
            json.dumps({"type": "answer", "data": "Ciao"}),
            json.dumps({"type": "answer", "data": " mondo"}),
        ]
        response = (line.encode() + b"\n" for line in payload)
        config = runner.BenchmarkConfig(
            root=runner.REPO_ROOT,
            url="http://example.test",
            target_paths=(),
            extensions=frozenset(),
            skip_dirs=frozenset(),
            use_planner=True,
            use_reasoning=False,
            timeout=1,
            chunk_size=1200,
            chunk_overlap=150,
            output_limit=100,
            error_limit=100,
            static_only=False,
        )
        answer, sources, reasoning, errors = runner.parse_stream(response, config)
        self.assertEqual("Ciao mondo", answer)
        self.assertEqual([{"filename": "a.py"}], sources)
        self.assertEqual([], reasoning)
        self.assertEqual([], errors)

    def test_static_metrics_with_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "backend").mkdir()
            (root / "backend" / "api.py").write_text("from fastapi import FastAPI\napp=FastAPI()\n@app.get('/x')\ndef x(): return 1\n", encoding="utf-8")
            (root / "frontend").mkdir()
            (root / "frontend" / "page.html").write_text("<script>fetch('/api/x')</script>", encoding="utf-8")
            config = runner.BenchmarkConfig(
                root=root,
                url="http://example.test",
                target_paths=("backend", "frontend"),
                extensions=frozenset({".py", ".html"}),
                skip_dirs=frozenset({"__pycache__"}),
                use_planner=True,
                use_reasoning=False,
                timeout=1,
                chunk_size=1200,
                chunk_overlap=150,
                output_limit=100,
                error_limit=100,
                static_only=False,
            )
            metrics = runner.static_metrics(config)
        self.assertEqual(2, metrics["files"])
        self.assertEqual(1, metrics["python_files"])
        self.assertEqual(1, metrics["fastapi_routes"][0]["path"] and len(metrics["fastapi_routes"]))
        self.assertEqual("/api/x", metrics["frontend_fetches"][0]["url"])

    def test_runtime_metrics_partial_and_failed(self) -> None:
        answers = [
            {"area": "a", "status": "ok", "latency_seconds": 1.0, "sources_count": 2},
            {"area": "a", "status": "partial", "latency_seconds": 3.0, "sources_count": 0},
            {"area": "b", "status": "failed", "latency_seconds": 0.5, "sources_count": 0},
        ]
        metrics = runner.runtime_metrics(answers)
        self.assertEqual(3, metrics["questions"])
        self.assertEqual(1, metrics["completed"])
        self.assertEqual(1, metrics["partial"])
        self.assertEqual(1, metrics["failed"])
        self.assertAlmostEqual(1.5, metrics["avg_latency_seconds"])
        self.assertEqual(1, metrics["answers_with_sources"])


if __name__ == "__main__":
    unittest.main()
