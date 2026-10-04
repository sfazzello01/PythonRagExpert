"""Runtime access to the static symbol graph created during indexing."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path

from config import (
    SYMBOL_GRAPH_FILE,
    SYMBOL_GRAPH_MAX_DEPTH,
    SYMBOL_GRAPH_PAGERANK_DAMPING,
    SYMBOL_GRAPH_PAGERANK_ITERATIONS,
    SYMBOL_GRAPH_PAGERANK_TOLERANCE,
    SYMBOL_GRAPH_VERSION,
)

logger = logging.getLogger("EspertoSymbolGraph")


class SymbolGraph:
    """Resolve related callers and callees to their enclosing indexed chunks."""

    def __init__(self, graph_path: Path = SYMBOL_GRAPH_FILE):
        self._graph_path = graph_path
        self._modified_at: int | None = None
        self._chunk_to_symbols: dict[str, set[str]] = {}
        self._nodes: dict[str, dict] = {}
        self._neighbors: dict[str, list[tuple[str, str]]] = {}
        self._out_edges: dict[str, list[str]] = {}

    def related_chunks(self, chunk_ids: set[str], limit: int) -> dict[str, str]:
        """Return graph-related chunks ranked by Personalized PageRank.

        I candidati sono raccolti con BFS bidirezionale entro la profondità
        configurata; il ranking usa PPR sulla sola adiacenza diretta
        (calls/inherits, source -> target). I chunk seed sono esclusi.
        """
        self._reload_if_needed()
        seed_symbols = {
            symbol_id
            for chunk_id in chunk_ids
            for symbol_id in self._chunk_to_symbols.get(chunk_id, set())
        }
        if not seed_symbols:
            return {}

        relation_paths = self._bfs_relation_paths(seed_symbols)
        ranks = self._personalized_pagerank(seed_symbols, set(relation_paths))

        best_by_chunk: dict[str, tuple[float, str]] = {}
        for symbol_id, score in ranks.items():
            if symbol_id in seed_symbols:
                continue
            chunk_id = self._chunk_for_symbol(symbol_id)
            if not chunk_id or chunk_id in chunk_ids:
                continue
            relation = relation_paths.get(symbol_id, "graph")
            previous = best_by_chunk.get(chunk_id)
            if previous is None or score > previous[0]:
                best_by_chunk[chunk_id] = (score, relation)

        # A parita' di score (es. i chiamanti, che il PPR diretto non raggiunge)
        # vince il percorso piu' corto dai seed.
        ordered = sorted(
            best_by_chunk,
            key=lambda chunk_id: (
                -best_by_chunk[chunk_id][0],
                best_by_chunk[chunk_id][1].count("->"),
                chunk_id,
            ),
        )
        return {chunk_id: best_by_chunk[chunk_id][1] for chunk_id in ordered[:limit]}

    def _bfs_relation_paths(self, seed_symbols: set[str]) -> dict[str, str]:
        """Collect candidate symbols within the configured depth and their shortest paths."""
        paths: dict[str, str] = {}
        queue = deque((symbol_id, 0, "") for symbol_id in sorted(seed_symbols))
        visited = set(seed_symbols)
        while queue:
            symbol_id, depth, relation_path = queue.popleft()
            if depth >= SYMBOL_GRAPH_MAX_DEPTH:
                continue
            for neighbor_id, relation in self._neighbors.get(symbol_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                next_path = (
                    f"{relation_path}->{relation}" if relation_path else relation
                )
                paths[neighbor_id] = next_path
                if depth + 1 < SYMBOL_GRAPH_MAX_DEPTH:
                    queue.append((neighbor_id, depth + 1, next_path))
        return paths

    def _personalized_pagerank(
        self, seed_symbols: set[str], candidates: set[str]
    ) -> dict[str, float]:
        """Rank symbols with Personalized PageRank over directed edges only."""
        nodes = sorted(seed_symbols | candidates)
        node_set = set(nodes)
        out_edges = {
            node_id: [
                target
                for target in self._out_edges.get(node_id, [])
                if target in node_set
            ]
            for node_id in nodes
        }
        personalization = 1.0 / len(seed_symbols)
        ranks = {
            node_id: personalization if node_id in seed_symbols else 0.0
            for node_id in nodes
        }
        damping = SYMBOL_GRAPH_PAGERANK_DAMPING
        for _ in range(SYMBOL_GRAPH_PAGERANK_ITERATIONS):
            next_ranks = {node_id: 0.0 for node_id in nodes}
            dangling_mass = 0.0
            for node_id in nodes:
                targets = out_edges[node_id]
                if targets:
                    share = damping * ranks[node_id] / len(targets)
                    for target in targets:
                        next_ranks[target] += share
                else:
                    dangling_mass += damping * ranks[node_id]
            # Teleport e massa dei nodi senza archi uscenti tornano ai soli seed:
            # la massa totale resta 1.
            for node_id in seed_symbols:
                next_ranks[node_id] += ((1.0 - damping) + dangling_mass) * personalization
            delta = sum(abs(next_ranks[node_id] - ranks[node_id]) for node_id in nodes)
            ranks = next_ranks
            if delta < SYMBOL_GRAPH_PAGERANK_TOLERANCE:
                break
        return ranks

    def _reload_if_needed(self) -> None:
        try:
            modified_at = self._graph_path.stat().st_mtime_ns
        except OSError:
            self._reset()
            return
        if modified_at == self._modified_at:
            return
        try:
            graph = json.loads(self._graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Cannot load symbol graph: %s", error)
            self._reset()
            return
        if graph.get("version") != SYMBOL_GRAPH_VERSION:
            logger.warning("Ignoring unsupported symbol graph version.")
            self._reset()
            return

        self._nodes = {node["id"]: node for node in graph.get("nodes", [])}
        chunk_to_symbols: dict[str, set[str]] = defaultdict(set)
        for node in self._nodes.values():
            if chunk_id := self._chunk_for_symbol(node["id"]):
                chunk_to_symbols[chunk_id].add(node["id"])
        neighbors: dict[str, list[tuple[str, str]]] = defaultdict(list)
        out_edges: dict[str, list[str]] = defaultdict(list)
        for edge in graph.get("edges", []):
            if edge.get("kind") not in {"calls", "inherits"}:
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source in self._nodes and target in self._nodes:
                neighbors[source].append((target, edge["kind"]))
                neighbors[target].append((source, f"called_by:{edge['kind']}"))
                out_edges[source].append(target)
        self._chunk_to_symbols = dict(chunk_to_symbols)
        self._neighbors = dict(neighbors)
        self._out_edges = dict(out_edges)
        self._modified_at = modified_at

    def _chunk_for_symbol(self, symbol_id: str) -> str | None:
        current = symbol_id
        while current:
            node = self._nodes.get(current)
            if node and node.get("chunk_id"):
                return node["chunk_id"]
            current = current.rsplit("::", 1)[0] if "::" in current else ""
        return None

    def _reset(self) -> None:
        self._modified_at = None
        self._chunk_to_symbols = {}
        self._nodes = {}
        self._neighbors = {}
        self._out_edges = {}