"""Unit test per la selezione graph: BFS bounded + Personalized PageRank diretto."""

import json
from pathlib import Path

from src.core.services.symbol_graph import SymbolGraph


def _node(symbol_id: str, chunk_id: str | None) -> dict:
    return {
        "id": symbol_id,
        "kind": "function",
        "name": symbol_id.rsplit("::", 1)[-1],
        "filename": "m.py",
        "start_line": 1,
        "end_line": 10,
        "chunk_id": chunk_id,
    }


def _write_graph(tmp_path: Path, edges: list[tuple[str, str]], version: int = 1) -> Path:
    graph_path = tmp_path / "symbol_graph.json"
    symbols = sorted({symbol for edge in edges for symbol in edge} | {"m::seed"})
    graph_path.write_text(
        json.dumps(
            {
                "version": version,
                "nodes": [
                    _node(symbol, f"chunk_{symbol.rsplit('::', 1)[-1]}")
                    for symbol in symbols
                ],
                "edges": [
                    {"source": source, "target": target, "kind": "calls", "line": 1}
                    for source, target in edges
                ],
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def test_bfs_stops_at_max_depth(tmp_path):
    graph_path = _write_graph(
        tmp_path,
        [
            ("m::seed", "m::a"),
            ("m::a", "m::b"),
            ("m::b", "m::c"),
            ("m::c", "m::d"),
        ],
    )
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert list(related) == ["chunk_a", "chunk_b", "chunk_c"]


def test_seeds_are_excluded(tmp_path):
    graph_path = _write_graph(
        tmp_path,
        [("m::seed", "m::a"), ("m::a", "m::seed")],
    )
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert "chunk_seed" not in related
    assert list(related) == ["chunk_a"]


def test_limit_is_applied_after_pagerank(tmp_path):
    edges = [("m::seed", f"m::t{index}") for index in range(1, 9)]
    graph_path = _write_graph(tmp_path, edges)
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert len(related) == 6
    assert list(related) == [f"chunk_t{index}" for index in range(1, 7)]


def test_pagerank_uses_only_directed_edges(tmp_path):
    # x -> seed -> z: la BFS trova sia x (called_by) sia z (calls),
    # ma il PPR propaga solo su seed -> z.
    graph_path = _write_graph(
        tmp_path,
        [("m::x", "m::seed"), ("m::seed", "m::z")],
    )
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert list(related) == ["chunk_z", "chunk_x"]


def test_shortest_bfs_path_is_exposed_as_relation(tmp_path):
    graph_path = _write_graph(
        tmp_path,
        [("m::seed", "m::a"), ("m::seed", "m::b"), ("m::b", "m::a")],
    )
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert related["chunk_a"] == "calls"


def test_missing_file_returns_empty(tmp_path):
    assert SymbolGraph(tmp_path / "absent.json").related_chunks({"chunk_seed"}, 6) == {}


def test_wrong_version_returns_empty(tmp_path):
    graph_path = _write_graph(tmp_path, [("m::seed", "m::a")], version=999)
    assert SymbolGraph(graph_path).related_chunks({"chunk_seed"}, 6) == {}


def test_seed_without_neighbors_returns_empty(tmp_path):
    graph_path = _write_graph(tmp_path, [])
    assert SymbolGraph(graph_path).related_chunks({"chunk_seed"}, 6) == {}


def test_unknown_seed_chunk_returns_empty(tmp_path):
    graph_path = _write_graph(tmp_path, [("m::seed", "m::a")])
    assert SymbolGraph(graph_path).related_chunks({"chunk_unknown"}, 6) == {}
