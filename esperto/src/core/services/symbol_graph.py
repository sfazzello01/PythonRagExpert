"""Accesso a runtime al symbol graph statico creato dall'indicizzazione."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from collections.abc import Sequence
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

# Oltre questo numero di definizioni un nome (es. `run`, `__init__`) è ambiguo e non viene usato.
MAX_NAME_MATCHES = 3
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


class SymbolGraph:
    """Trova i chunk collegati nel grafo (chiamanti, chiamati, basi) e quelli dei simboli nominati."""

    def __init__(self, graph_path: Path = SYMBOL_GRAPH_FILE):
        """Crea l'accesso al grafo; il file viene letto alla prima interrogazione.

        Args:
            graph_path: File JSON del symbol graph.
        """
        self._graph_path = graph_path
        self._modified_at: int | None = None
        self._chunk_to_symbols: dict[str, set[str]] = {}
        self._nodes: dict[str, dict] = {}
        self._neighbors: dict[str, list[tuple[str, str]]] = {}
        self._out_edges: dict[str, list[str]] = {}
        self._chunks_by_name: dict[str, set[str]] = {}

    def chunks_named(self, text: str) -> list[str]:
        """Trova i chunk che definiscono i simboli nominati in un testo.

        Contano solo le parole con la forma di un identificatore (con maiuscole, `_` o punti); un nome
        definito in più di MAX_NAME_MATCHES chunk è ambiguo e viene ignorato.

        Args:
            text: Domanda in cui cercare i nomi.

        Returns:
            Gli ID dei chunk, senza duplicati, nell'ordine in cui i nomi compaiono nel testo.
        """
        self._reload_if_needed()
        chunk_ids: dict[str, None] = {}
        for token in _IDENTIFIER.findall(text):
            # Solo le parole che hanno la forma di un identificatore: una parola comune in
            # minuscolo ("query", "test") non deve agganciare un simbolo omonimo.
            if token == token.lower() and "_" not in token and "." not in token:
                continue
            for name in {token, *token.split(".")}:
                matches = self._chunks_by_name.get(name, ())
                if len(matches) <= MAX_NAME_MATCHES:
                    chunk_ids.update(dict.fromkeys(sorted(matches)))
        return list(chunk_ids)

    def related_chunks(self, chunk_ids: set[str], limit: int, top_chunk_ids: Sequence[str] = ()) -> dict[str, str]:
        """Trova i chunk collegati ai semi nel grafo, ordinati con Personalized PageRank.

        I candidati sono raccolti con una BFS bidirezionale entro la profondità configurata; l'ordine usa il
        PageRank sulla sola adiacenza diretta (`calls`/`inherits`, da sorgente a destinazione). I vicini a
        un passo di `top_chunk_ids` precedono tutti gli altri.

        Args:
            chunk_ids: Chunk seme, esclusi dal risultato.
            limit: Numero massimo di chunk.
            top_chunk_ids: Primi risultati, i cui vicini diretti hanno la precedenza.

        Returns:
            Per ogni chunk collegato, in ordine di priorità, il percorso di relazioni che lo lega ai semi.
        """
        self._reload_if_needed()
        seed_symbols = {
            symbol_id for chunk_id in chunk_ids for symbol_id in self._chunk_to_symbols.get(chunk_id, set())
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

        # I vicini diretti dei primi risultati completano le catene di chiamate: vengono prima,
        # e tra loro prima quelli nello stesso scope (classe o modulo) del risultato.
        priority: dict[str | None, int] = {}
        for chunk_id in top_chunk_ids:
            for symbol_id in self._chunk_to_symbols.get(chunk_id, ()):
                scope = symbol_id.rpartition("::")[0]
                for neighbor_id, _ in self._neighbors.get(symbol_id, []):
                    neighbor_chunk = self._chunk_for_symbol(neighbor_id)
                    tier = 0 if neighbor_id.rpartition("::")[0] == scope else 1
                    priority[neighbor_chunk] = min(tier, priority.get(neighbor_chunk, tier))

        # A parità di score (es. i chiamanti, che il PPR diretto non raggiunge)
        # vince il percorso più corto dai seed.
        ordered = sorted(
            best_by_chunk,
            key=lambda chunk_id: (
                priority.get(chunk_id, 2),
                -best_by_chunk[chunk_id][0],
                best_by_chunk[chunk_id][1].count("->"),
                chunk_id,
            ),
        )
        return {chunk_id: best_by_chunk[chunk_id][1] for chunk_id in ordered[:limit]}

    def _bfs_relation_paths(self, seed_symbols: set[str]) -> dict[str, str]:
        """Raccoglie i simboli raggiungibili dai semi entro la profondità configurata.

        Args:
            seed_symbols: Simboli di partenza.

        Returns:
            Per ogni simbolo raggiunto, il percorso di relazioni più breve che lo lega a un seme.
        """
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
                next_path = f"{relation_path}->{relation}" if relation_path else relation
                paths[neighbor_id] = next_path
                if depth + 1 < SYMBOL_GRAPH_MAX_DEPTH:
                    queue.append((neighbor_id, depth + 1, next_path))
        return paths

    def _personalized_pagerank(self, seed_symbols: set[str], candidates: set[str]) -> dict[str, float]:
        """Ordina i simboli con Personalized PageRank, seguendo i soli archi orientati.

        Args:
            seed_symbols: Simboli a cui torna il teletrasporto.
            candidates: Simboli raggiunti dalla BFS.

        Returns:
            Lo score di ogni simbolo, semi compresi; la somma degli score è 1.
        """
        nodes = sorted(seed_symbols | candidates)
        node_set = set(nodes)
        out_edges = {
            node_id: [target for target in self._out_edges.get(node_id, []) if target in node_set]
            for node_id in nodes
        }
        personalization = 1.0 / len(seed_symbols)
        ranks = {node_id: personalization if node_id in seed_symbols else 0.0 for node_id in nodes}
        damping = SYMBOL_GRAPH_PAGERANK_DAMPING
        for _ in range(SYMBOL_GRAPH_PAGERANK_ITERATIONS):
            next_ranks = dict.fromkeys(nodes, 0.0)
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
        """Ricarica il grafo se il file è cambiato dall'ultima lettura; lo svuota se manca o non è valido."""
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
            logger.warning("Impossibile caricare il symbol graph: %s", error)
            self._reset()
            return
        if graph.get("version") != SYMBOL_GRAPH_VERSION:
            logger.warning("Versione del symbol graph non supportata: grafo ignorato.")
            self._reset()
            return

        self._nodes = {node["id"]: node for node in graph.get("nodes", [])}
        chunk_to_symbols: dict[str, set[str]] = defaultdict(set)
        chunks_by_name: dict[str, set[str]] = defaultdict(set)
        for node in self._nodes.values():
            if chunk_id := self._chunk_for_symbol(node["id"]):
                chunk_to_symbols[chunk_id].add(node["id"])
                # Tabella dei simboli: nome corto ("login") e qualificato ("BaseUser.login").
                chunks_by_name[node["name"]].add(chunk_id)
                chunks_by_name[node["id"].split("::", 1)[-1].replace("::", ".")].add(chunk_id)
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
        self._chunks_by_name = dict(chunks_by_name)
        self._neighbors = dict(neighbors)
        self._out_edges = dict(out_edges)
        self._modified_at = modified_at

    def _chunk_for_symbol(self, symbol_id: str) -> str | None:
        """Risale da un simbolo al chunk che lo contiene.

        Args:
            symbol_id: ID del simbolo.

        Returns:
            L'ID del chunk, oppure None se nessun simbolo che lo contiene ha un chunk.
        """
        current = symbol_id
        while current:
            node = self._nodes.get(current)
            if node and node.get("chunk_id"):
                return node["chunk_id"]
            current = current.rpartition("::")[0]
        return None

    def _reset(self) -> None:
        """Svuota il grafo in memoria."""
        self._modified_at = None
        self._chunk_to_symbols = {}
        self._chunks_by_name = {}
        self._nodes = {}
        self._neighbors = {}
        self._out_edges = {}
