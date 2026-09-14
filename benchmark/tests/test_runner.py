"""Test isolati del benchmark esterno."""

from __future__ import annotations

import json
import unittest

from benchmark import runner


class RunnerTest(unittest.TestCase):
    def test_queries_schema_and_count(self) -> None:
        queries = runner.load_queries(runner.ROOT / "queries.json")
        self.assertEqual(50, len(queries))
        self.assertEqual(50, len({item["id"] for item in queries}))
        self.assertTrue(all(item["area"] and item["question"] for item in queries))

    def test_normalize_source_path_removes_historical_prefix(self) -> None:
        self.assertEqual(
            "backend/src/server_api/api_server.py",
            runner.normalize_source_path("labnew/backend/src/server_api/api_server.py#api_parse"),
        )

    def test_source_matches_filename_and_suffix(self) -> None:
        self.assertTrue(
            runner.source_matches(
                "backend/src/server_api/api_server.py",
                {"filename": "labnew/backend/src/server_api/api_server.py", "symbol": "parse"},
            )
        )
        self.assertFalse(
            runner.source_matches(
                "backend/src/server_api/api_server.py",
                {"filename": "labnew/frontend/src/main.py"},
            )
        )

    def test_retrieval_metrics_deduplicate_and_use_ranking(self) -> None:
        expected = ["backend/src/a.py", "backend/src/b.py"]
        returned = [
            {"filename": "labnew/backend/src/no.py"},
            {"filename": "labnew/backend/src/a.py"},
            {"filename": "labnew/backend/src/a.py"},
            {"filename": "labnew/backend/src/b.py"},
        ]
        metrics = runner.retrieval_metrics(expected, returned)
        self.assertEqual(0.6667, metrics["precision_at_5"])
        self.assertEqual(1.0, metrics["recall_at_5"])
        self.assertEqual(1, metrics["hit_at_5"])
        self.assertEqual(0.5, metrics["mrr"])
        self.assertEqual(3, metrics["unique_returned_sources"])

    def test_retrieval_metrics_without_sources(self) -> None:
        metrics = runner.retrieval_metrics(["backend/src/a.py"], [])
        self.assertEqual(0.0, metrics["precision_at_5"])
        self.assertEqual(0.0, metrics["recall_at_5"])
        self.assertEqual(0, metrics["hit_at_5"])
        self.assertEqual(0.0, metrics["mrr"])

    def test_retrieval_metrics_without_expected_sources_is_not_evaluable(self) -> None:
        metrics = runner.retrieval_metrics([], [{"filename": "backend/src/a.py"}])
        self.assertIsNone(metrics["recall_at_5"])
        self.assertEqual(0, metrics["hit_at_5"])

    def test_aggregate_retrieval_by_area_and_median(self) -> None:
        answers = [
            {"area": "backend", "retrieval_metrics": {"precision_at_5": 1.0, "recall_at_5": 0.5, "hit_at_5": 1, "mrr": 1.0}},
            {"area": "backend", "retrieval_metrics": {"precision_at_5": 0.0, "recall_at_5": 0.0, "hit_at_5": 0, "mrr": 0.0}},
            {"area": "frontend", "retrieval_metrics": {"precision_at_5": 0.5, "recall_at_5": 1.0, "hit_at_5": 1, "mrr": 0.5}},
        ]
        metrics = runner.aggregate_retrieval(answers)
        self.assertEqual(3, metrics["questions"])
        self.assertEqual(0.5, metrics["median_precision_at_5"])
        self.assertEqual(0.5, metrics["by_area"]["backend"]["avg_precision_at_5"])

    def test_parse_stream(self) -> None:
        payload = [
            json.dumps({"type": "sources", "data": {"sources": [{"filename": "a.py"}]}}),
            json.dumps({"type": "answer", "data": "Ciao"}),
            json.dumps({"type": "answer", "data": " mondo"}),
        ]
        response = (line.encode() + b"\n" for line in payload)
        config = runner.BenchmarkConfig(
            url="http://example.test",
            use_planner=True,
            use_reasoning=False,
            timeout=1,
            output_limit=100,
            error_limit=100,
        )
        answer, sources, reasoning, errors = runner.parse_stream(response, config)
        self.assertEqual("Ciao mondo", answer)
        self.assertEqual([{"filename": "a.py"}], sources)
        self.assertEqual([], reasoning)
        self.assertEqual([], errors)

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
