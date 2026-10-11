"""Costruzione statica del grafo dei simboli e delle chiamate del codice Python indicizzato."""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from config import SYMBOL_GRAPH_FILE, SYMBOL_GRAPH_VERSION

from .chunking import splits_into_methods, symbol_chunk_id
from .discovery import relative_path

logger = logging.getLogger("EspertoSymbolGraph")

# Chiamate elencate al massimo nell'intestazione di un chunk.
MAX_HEADER_CALLS = 8
# Riesportazioni seguite al massimo per arrivare alla definizione di un nome importato.
MAX_REEXPORTS = 5


@dataclass(frozen=True)
class SymbolNode:
    """Modulo, classe, funzione o metodo definito nella codebase indicizzata."""

    id: str
    kind: str
    name: str
    filename: str
    start_line: int
    end_line: int
    chunk_id: str | None


@dataclass(frozen=True)
class SymbolEdge:
    """Relazione statica orientata tra due simboli."""

    source: str
    target: str
    kind: str
    line: int


def _module_name(path: Path) -> str:
    """Ricava il nome puntato del modulo di un file Python.

    Args:
        path: Percorso del file.

    Returns:
        Il nome del modulo; per un `__init__.py`, quello del package.
    """
    relative = relative_path(path)
    if relative.endswith("/__init__.py"):
        relative = relative[: -len("/__init__.py")]
    elif relative.endswith(".py"):
        relative = relative[:-3]
    return relative.replace("/", ".")


def _import_module(current: str, is_package: bool, module: str | None, level: int) -> str:
    """Risolve il nome assoluto del modulo indicato da un `from ... import`, anche relativo.

    Args:
        current: Nome del modulo che contiene l'import.
        is_package: True se il modulo corrente è un `__init__.py`.
        module: Modulo scritto nell'istruzione; None per `from . import nome`.
        level: Numero di punti dell'import relativo, 0 se è assoluto.

    Returns:
        Il nome assoluto del modulo importato.
    """
    if not level:
        return module or ""
    package = current.split(".") if is_package else current.split(".")[:-1]
    parent = package[: max(0, len(package) - level + 1)]
    return ".".join((*parent, *(module.split(".") if module else ())))


def _attribute_name(node: ast.AST) -> str | None:
    """Ricostruisce il nome puntato di un riferimento come `modulo.Classe.metodo`.

    Args:
        node: Espressione da cui ricavare il nome.

    Returns:
        Il nome puntato, oppure None se l'espressione non è una catena di nomi e attributi.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


class _GraphVisitor(ast.NodeVisitor):
    """Visita un modulo raccogliendone simboli e relazioni: definizioni, import, chiamate, ereditarietà."""

    def __init__(self, filename: str, module: str, is_package: bool, definitions: set[str], aliases: dict[str, str]):
        """Prepara la visita di un modulo.

        Args:
            filename: Percorso relativo del file.
            module: Nome puntato del modulo.
            is_package: True se il file è un `__init__.py`.
            definitions: ID di tutti i simboli definiti nella codebase.
            aliases: Nomi riesportati dai moduli, ciascuno con il simbolo a cui rimanda.
        """
        self.filename = filename
        self.module = module
        self.is_package = is_package
        self.definitions = definitions
        self.aliases = aliases
        self.nodes: list[SymbolNode] = []
        self.edges: list[SymbolEdge] = []
        self.scope: list[str] = []
        self.split_classes: set[str] = set()
        self.imports: dict[str, str] = {}

    @property
    def current_id(self) -> str:
        """ID del simbolo in cui si trova la visita: il modulo seguito dallo scope corrente."""
        return "::".join((self.module, *self.scope))

    def visit_Module(self, node: ast.Module) -> None:
        """Registra il nodo del modulo e ne visita il contenuto.

        Args:
            node: Modulo da visitare.
        """
        self.nodes.append(
            SymbolNode(
                id=self.module,
                kind="module",
                name=self.module.rsplit(".", 1)[-1],
                filename=self.filename,
                start_line=1,
                end_line=max(1, getattr(node, "end_lineno", 1) or 1),
                chunk_id=None,
            )
        )
        self.generic_visit(node)

    def _add_symbol(self, node: ast.AST, kind: str, name: str) -> str:
        """Registra un simbolo e l'arco `defines` che lo lega allo scope corrente.

        Args:
            node: Definizione del simbolo.
            kind: Tipo di simbolo: `class`, `function` o `method`.
            name: Nome del simbolo.

        Returns:
            L'ID del simbolo registrato.
        """
        symbol_id = "::".join((self.module, *self.scope, name))
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line) or start_line
        self.nodes.append(
            SymbolNode(
                id=symbol_id,
                kind=kind,
                name=name,
                filename=self.filename,
                start_line=start_line,
                end_line=end_line,
                chunk_id=self._chunk_id(node, name),
            )
        )
        self.edges.append(SymbolEdge(self.current_id, symbol_id, "defines", start_line))
        return symbol_id

    def _chunk_id(self, node: ast.AST, name: str) -> str | None:
        """Calcola l'ID del chunk che contiene il simbolo.

        Hanno un chunk proprio i simboli di primo livello e i metodi delle classi spezzate.

        Args:
            node: Definizione del simbolo.
            name: Nome del simbolo.

        Returns:
            L'ID del chunk, oppure None se il simbolo sta nel chunk di un simbolo che lo contiene.
        """
        if not self.scope:
            return symbol_chunk_id(self.filename, name, node)
        is_method = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(self.scope) == 1
        if is_method and self.scope[0] in self.split_classes:
            return symbol_chunk_id(self.filename, f"{self.scope[0]}.{name}", node)
        return None

    def _visit_definition(self, node: ast.AST, kind: str, name: str) -> None:
        """Registra una definizione e ne visita il corpo con lo scope aggiornato.

        Args:
            node: Definizione da visitare.
            kind: Tipo di simbolo: `function` o `method`.
            name: Nome del simbolo.
        """
        self._add_symbol(node, kind, name)
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Registra una funzione o un metodo, anche asincroni, e ne visita il corpo.

        Args:
            node: Definizione da visitare.
        """
        self._visit_definition(node, "function" if not self.scope else "method", node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Registra una classe con gli archi `inherits` verso le sue basi e ne visita il corpo.

        Args:
            node: Definizione della classe.
        """
        if not self.scope and splits_into_methods(node):
            self.split_classes.add(node.name)
        class_id = self._add_symbol(node, "class", node.name)
        for base in node.bases:
            if isinstance(base, ast.Subscript):  # base generica: Engine[PostgresTransaction]
                base = base.value
            target = self._resolve_reference(_attribute_name(base))
            if target:
                self.edges.append(SymbolEdge(class_id, target, "inherits", node.lineno))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        """Annota i nomi importati con `import` e l'arco verso i moduli della codebase.

        Args:
            node: Istruzione `import`.
        """
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", 1)[0]
            self.imports[local_name] = alias.name
            if alias.name in self.definitions:
                self.edges.append(SymbolEdge(self.current_id, alias.name, "imports", node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Annota i nomi importati con `from ... import` e l'arco verso moduli e simboli della codebase.

        Args:
            node: Istruzione `from ... import`.
        """
        module = _import_module(self.module, self.is_package, node.module, node.level)
        for alias in node.names:
            local_name = alias.asname or alias.name
            target = f"{module}::{alias.name}" if module else alias.name
            self.imports[local_name] = target
            import_target = module if module in self.definitions else target
            if import_target in self.definitions:
                self.edges.append(SymbolEdge(self.current_id, import_target, "imports", node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        """Aggiunge l'arco `calls` se la chiamata punta a un simbolo della codebase.

        Args:
            node: Chiamata da analizzare.
        """
        target = self._resolve_reference(_attribute_name(node.func))
        if target:
            self.edges.append(SymbolEdge(self.current_id, target, "calls", node.lineno))
        self.generic_visit(node)

    def _definition(self, candidate: str) -> str | None:
        """Risolve un candidato nel simbolo che lo definisce, seguendo le riesportazioni.

        Un nome importato in un `__init__.py` rimanda al modulo che lo definisce davvero.

        Args:
            candidate: ID del simbolo cercato.

        Returns:
            L'ID della definizione, oppure None se il simbolo non appartiene alla codebase.
        """
        for _ in range(MAX_REEXPORTS):
            if candidate in self.definitions:
                return candidate
            parts = candidate.split("::")
            target = self.aliases.get("::".join(parts[:2]))
            if not target:
                return None
            candidate = "::".join((target, *parts[2:]))
        return None

    def _resolve_reference(self, reference: str | None) -> str | None:
        """Risolve un riferimento scritto nel codice nel simbolo della codebase a cui punta.

        Args:
            reference: Nome puntato come appare nel sorgente (`self.metodo`, `modulo.funzione`).

        Returns:
            L'ID del simbolo, oppure None se il riferimento è vuoto o esterno alla codebase.
        """
        if not reference:
            return None
        parts = reference.split(".")
        # `self.metodo()` dentro un metodo: il simbolo si cerca nella classe che lo contiene.
        if parts[0] in {"self", "cls"} and len(parts) > 1 and len(self.scope) > 1:
            return self._definition(f"{self.module}::{self.scope[-2]}::{parts[-1]}")
        imported = self.imports.get(parts[0])
        if imported:
            return self._definition("::".join((imported, *parts[1:])))
        return self._definition("::".join((self.module, *parts)))


class _DefinitionVisitor(ast.NodeVisitor):
    """Raccoglie gli ID delle definizioni e le riesportazioni di un modulo, prima di risolvere gli archi."""

    def __init__(self, module: str, is_package: bool):
        """Prepara la visita di un modulo.

        Args:
            module: Nome puntato del modulo.
            is_package: True se il file è un `__init__.py`.
        """
        self.module = module
        self.is_package = is_package
        self.scope: list[str] = []
        self.definitions = {module}
        # "modulo::nome" -> "modulo di origine::nome" per i nomi importati a livello di modulo.
        self.aliases: dict[str, str] = {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Annota i nomi importati a livello di modulo come possibili riesportazioni.

        Args:
            node: Istruzione `from ... import`.
        """
        origin = _import_module(self.module, self.is_package, node.module, node.level)
        if origin and not self.scope:
            for alias in node.names:
                self.aliases[f"{self.module}::{alias.asname or alias.name}"] = f"{origin}::{alias.name}"

    def _visit_definition(self, node: ast.AST, name: str) -> None:
        """Registra l'ID di una definizione e ne visita il corpo.

        Args:
            node: Definizione da visitare.
            name: Nome del simbolo.
        """
        self.definitions.add("::".join((self.module, *self.scope, name)))
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Registra una funzione o un metodo, anche asincroni.

        Args:
            node: Definizione da visitare.
        """
        self._visit_definition(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Registra una classe.

        Args:
            node: Definizione della classe.
        """
        self._visit_definition(node, node.name)


def _parse(files: list[Path]) -> dict[Path, ast.Module]:
    """Analizza i file Python della codebase.

    Args:
        files: File della codebase; quelli non Python sono ignorati.

    Returns:
        L'AST di ogni file; un file illeggibile o con errori di sintassi viene saltato e segnalato.
    """
    trees: dict[Path, ast.Module] = {}
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError) as error:
            logger.warning("Impossibile analizzare %s per il symbol graph: %s", path, error)
    return trees


def build_symbol_graph(files: list[Path]) -> dict:
    """Costruisce il grafo statico dei file Python, senza importare il codice dell'applicazione.

    Args:
        files: File della codebase.

    Returns:
        Il grafo con versione, nodi (simboli) e archi (`defines`, `imports`, `calls`, `inherits`).
    """
    trees = _parse(files)
    # Primo passaggio: tutte le definizioni, necessarie per risolvere chiamate tra moduli.
    definitions: set[str] = set()
    aliases: dict[str, str] = {}
    for path, tree in trees.items():
        definition_visitor = _DefinitionVisitor(_module_name(path), path.name == "__init__.py")
        definition_visitor.visit(tree)
        definitions.update(definition_visitor.definitions)
        aliases.update(definition_visitor.aliases)
    nodes: list[SymbolNode] = []
    edges: list[SymbolEdge] = []
    for path, tree in trees.items():
        visitor = _GraphVisitor(
            relative_path(path), _module_name(path), path.name == "__init__.py", definitions, aliases
        )
        visitor.visit(tree)
        nodes.extend(visitor.nodes)
        edges.extend(visitor.edges)
    unique_edges = {(edge.source, edge.target, edge.kind, edge.line): edge for edge in edges}
    return {
        "version": SYMBOL_GRAPH_VERSION,
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in unique_edges.values()],
    }


def chunk_relations(graph: dict) -> dict[str, dict[str, list[str]]]:
    """Ricava, per ogni chunk, i simboli da cui eredita e quelli che chiama.

    Sono le relazioni scritte nell'intestazione del chunk.

    Args:
        graph: Symbol graph della codebase.

    Returns:
        Per ogni chunk_id, i nomi sotto le chiavi `inherits` e `calls` (al massimo MAX_HEADER_CALLS).
    """
    nodes = {node["id"]: node for node in graph["nodes"]}

    def chunk_of(symbol_id: str) -> str | None:
        """Risale da un simbolo al chunk che lo contiene.

        Args:
            symbol_id: ID del simbolo.

        Returns:
            L'ID del chunk, oppure None se nessun simbolo che lo contiene ha un chunk.
        """
        while symbol_id:
            if chunk_id := nodes.get(symbol_id, {}).get("chunk_id"):
                return chunk_id
            symbol_id = symbol_id.rpartition("::")[0]
        return None

    relations: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"inherits": [], "calls": []})
    class_bases: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge["kind"] not in ("calls", "inherits"):
            continue
        chunk_id = chunk_of(edge["source"])
        if not chunk_id or chunk_of(edge["target"]) == chunk_id:
            continue
        name = edge["target"].split("::", 1)[-1].replace("::", ".")
        names = relations[chunk_id][edge["kind"]]
        if name not in names:
            names.append(name)
        if edge["kind"] == "inherits":
            class_bases[edge["source"]] = relations[chunk_id]["inherits"]
    # I metodi di una classe spezzata stanno in chunk propri: riportano le basi della classe.
    for node in graph["nodes"]:
        bases = class_bases.get(node["id"].rpartition("::")[0])
        if bases and node["chunk_id"]:
            relations[node["chunk_id"]]["inherits"] = bases
    for related in relations.values():
        del related["calls"][MAX_HEADER_CALLS:]
    return dict(relations)


def save_symbol_graph(graph: dict) -> None:
    """Salva il grafo dei sorgenti correnti con una scrittura atomica.

    Args:
        graph: Symbol graph da salvare.
    """
    SYMBOL_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = SYMBOL_GRAPH_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_file.replace(SYMBOL_GRAPH_FILE)
    logger.info("Symbol graph salvato: %d nodi, %d archi.", len(graph["nodes"]), len(graph["edges"]))
