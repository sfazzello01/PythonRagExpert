"""Unit test per la selezione graph: BFS bounded + Personalized PageRank diretto."""

import json
from pathlib import Path

from src.core.indexing.symbol_graph import MAX_HEADER_CALLS, chunk_relations
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


def test_pagerank_conserves_mass(tmp_path):
    edges = [("m::seed", f"m::t{index}") for index in range(50)]
    graph = SymbolGraph(_write_graph(tmp_path, edges))
    graph._reload_if_needed()
    seeds = {"m::seed"}
    ranks = graph._personalized_pagerank(seeds, set(graph._bfs_relation_paths(seeds)))
    assert abs(sum(ranks.values()) - 1.0) < 1e-9


def test_equal_scores_prefer_the_shortest_path(tmp_path):
    # Il PPR diretto non raggiunge i chiamanti: tra a (2 hop) e z (1 hop) vince z.
    graph_path = _write_graph(tmp_path, [("m::a", "m::z"), ("m::z", "m::seed")])
    related = SymbolGraph(graph_path).related_chunks({"chunk_seed"}, limit=6)
    assert list(related) == ["chunk_z", "chunk_a"]


def _symbol_table(tmp_path: Path) -> SymbolGraph:
    nodes = [
        _node("m::BaseUser", "chunk_user"),
        _node("m::BaseUser::login", None),  # metodo di una classe non spezzata: sta nel chunk della classe
        _node("m::hash_password", "chunk_hash"),
        *(_node(f"m::Engine{index}::run", f"chunk_run{index}") for index in range(4)),
    ]
    graph_path = tmp_path / "symbol_graph.json"
    graph_path.write_text(json.dumps({"version": 1, "nodes": nodes, "edges": []}), encoding="utf-8")
    return SymbolGraph(graph_path)


def test_symbols_are_found_by_short_and_qualified_name(tmp_path):
    table = _symbol_table(tmp_path)
    assert table.chunks_named("Cosa fa `BaseUser.login`?") == ["chunk_user"]
    assert table.chunks_named("come funziona hash_password e BaseUser") == ["chunk_hash", "chunk_user"]


def test_ambiguous_and_plain_words_are_not_looked_up(tmp_path):
    table = _symbol_table(tmp_path)
    assert table.chunks_named("cosa fa self.run?") == []  # `run` e' definito in quattro chunk
    assert table.chunks_named("come funziona il login") == []  # parola comune, non un identificatore


def test_chunk_relations_list_bases_and_calls_per_chunk():
    nodes = [
        _node("m", None),
        _node("m::Base", "chunk_base"),
        _node("m::Child", "chunk_child"),
        _node("m::Child::run", "chunk_run"),  # classe spezzata: il metodo ha un chunk proprio
        _node("m::Child::helper", None),
        *(_node(f"m::f{index}", f"chunk_f{index}") for index in range(MAX_HEADER_CALLS + 2)),
    ]
    edges = [
        {"source": "m::Child", "target": "m::Base", "kind": "inherits", "line": 1},
        {"source": "m::Child::helper", "target": "m::Child::run", "kind": "calls", "line": 2},
        {"source": "m::Child::helper", "target": "m::Child::helper", "kind": "calls", "line": 3},
        {"source": "m", "target": "m::Base", "kind": "calls", "line": 4},  # codice di modulo: nessun chunk
        *(
            {"source": "m::Child::run", "target": f"m::f{index}", "kind": "calls", "line": 5}
            for index in range(MAX_HEADER_CALLS + 2)
        ),
    ]
    relations = chunk_relations({"nodes": nodes, "edges": edges})
    assert relations["chunk_child"] == {"inherits": ["Base"], "calls": ["Child.run"]}
    assert relations["chunk_run"]["inherits"] == ["Base"]
    assert relations["chunk_run"]["calls"] == [f"f{index}" for index in range(MAX_HEADER_CALLS)]
    assert "chunk_base" not in relations
