"""Runtime access to the static symbol graph created during indexing."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from ..shared.settings import SYMBOL_GRAPH_FILE, SYMBOL_GRAPH_VERSION

logger = logging.getLogger("EspertoSymbolGraph")


class SymbolGraph:
    """Resolve one-hop callers and callees to their enclosing indexed chunks."""

    def __init__(self, graph_path: Path = SYMBOL_GRAPH_FILE):
        self._graph_path = graph_path
        self._modified_at: int | None = None
        self._chunk_to_symbols: dict[str, set[str]] = {}
        self._nodes: dict[str, dict] = {}
        self._neighbors: dict[str, list[tuple[str, str]]] = {}

    def related_chunks(self, chunk_ids: set[str], limit: int) -> dict[str, str]:
        """Return directly connected chunk IDs mapped to their graph relation."""
        self._reload_if_needed()
        related: dict[str, str] = {}
        for chunk_id in chunk_ids:
            for symbol_id in self._chunk_to_symbols.get(chunk_id, set()):
                for neighbor_id, relation in self._neighbors.get(symbol_id, []):
                    neighbor_chunk_id = self._chunk_for_symbol(neighbor_id)
                    if neighbor_chunk_id and neighbor_chunk_id not in chunk_ids:
                        related.setdefault(neighbor_chunk_id, relation)
                        if len(related) >= limit:
                            return related
        return related

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
        for edge in graph.get("edges", []):
            if edge.get("kind") not in {"calls", "inherits"}:
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source in self._nodes and target in self._nodes:
                neighbors[source].append((target, edge["kind"]))
                neighbors[target].append((source, f"called_by:{edge['kind']}"))
        self._chunk_to_symbols = dict(chunk_to_symbols)
        self._neighbors = dict(neighbors)
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