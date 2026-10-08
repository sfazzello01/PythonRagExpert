"""Static Python symbol and call graph generation for indexed source code."""

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
    """A module, class, function, or method defined in the indexed codebase."""

    id: str
    kind: str
    name: str
    filename: str
    start_line: int
    end_line: int
    chunk_id: str | None


@dataclass(frozen=True)
class SymbolEdge:
    """A directed static relationship between two symbols."""

    source: str
    target: str
    kind: str
    line: int


def _module_name(path: Path) -> str:
    relative = relative_path(path)
    if relative.endswith("/__init__.py"):
        relative = relative[: -len("/__init__.py")]
    elif relative.endswith(".py"):
        relative = relative[:-3]
    return relative.replace("/", ".")


def _import_module(current: str, is_package: bool, module: str | None, level: int) -> str:
    """Absolute name of the module named by a (possibly relative) `from ... import`."""
    if not level:
        return module or ""
    package = current.split(".") if is_package else current.split(".")[:-1]
    parent = package[: max(0, len(package) - level + 1)]
    return ".".join((*parent, *(module.split(".") if module else ())))


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


class _GraphVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, module: str, is_package: bool, definitions: set[str], aliases: dict[str, str]):
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
        return "::".join((self.module, *self.scope))

    def visit_Module(self, node: ast.Module) -> None:
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
        """ID of the chunk holding the symbol: top-level symbols and methods of split classes."""
        if not self.scope:
            return symbol_chunk_id(self.filename, name, node)
        is_method = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(self.scope) == 1
        if is_method and self.scope[0] in self.split_classes:
            return symbol_chunk_id(self.filename, f"{self.scope[0]}.{name}", node)
        return None

    def _visit_definition(self, node: ast.AST, kind: str, name: str) -> None:
        self._add_symbol(node, kind, name)
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node, "function" if not self.scope else "method", node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
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
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", 1)[0]
            self.imports[local_name] = alias.name
            if alias.name in self.definitions:
                self.edges.append(SymbolEdge(self.current_id, alias.name, "imports", node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _import_module(self.module, self.is_package, node.module, node.level)
        for alias in node.names:
            local_name = alias.asname or alias.name
            target = f"{module}::{alias.name}" if module else alias.name
            self.imports[local_name] = target
            import_target = module if module in self.definitions else target
            if import_target in self.definitions:
                self.edges.append(SymbolEdge(self.current_id, import_target, "imports", node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        target = self._resolve_reference(_attribute_name(node.func))
        if target:
            self.edges.append(SymbolEdge(self.current_id, target, "calls", node.lineno))
        self.generic_visit(node)

    def _definition(self, candidate: str) -> str | None:
        """Return the candidate if it is defined, following re-exports (e.g. via `__init__.py`)."""
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
        if not reference:
            return None
        parts = reference.split(".")
        if parts[0] in {"self", "cls"} and len(parts) > 1 and self.scope:
            class_name = next((name for name in reversed(self.scope[:-1])), None)
            if class_name:
                candidate = f"{self.module}::{class_name}::{parts[-1]}"
                return self._definition(candidate)
        imported = self.imports.get(parts[0])
        if imported:
            candidate = "::".join((imported, *parts[1:])) if len(parts) > 1 else imported
            return self._definition(candidate)
        candidate = "::".join((self.module, *parts))
        return self._definition(candidate)


class _DefinitionVisitor(ast.NodeVisitor):
    """Collect fully-qualified definition IDs and module-level re-exports before resolving graph edges."""

    def __init__(self, module: str, is_package: bool):
        self.module = module
        self.is_package = is_package
        self.scope: list[str] = []
        self.definitions = {module}
        # "modulo::nome" -> "modulo di origine::nome" per i nomi importati a livello di modulo.
        self.aliases: dict[str, str] = {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        origin = _import_module(self.module, self.is_package, node.module, node.level)
        if origin and not self.scope:
            for alias in node.names:
                self.aliases[f"{self.module}::{alias.asname or alias.name}"] = f"{origin}::{alias.name}"

    def _visit_definition(self, node: ast.AST, name: str) -> None:
        self.definitions.add("::".join((self.module, *self.scope, name)))
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_definition(node, node.name)


def _parse(files: list[Path]) -> dict[Path, ast.Module]:
    trees: dict[Path, ast.Module] = {}
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError) as error:
            logger.warning("Cannot parse %s for symbol graph: %s", path, error)
    return trees


def build_symbol_graph(files: list[Path]) -> dict:
    """Build a static graph for Python files without importing application code."""
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
    """Per chunk, the symbols it inherits from and calls: they are written in the chunk header."""
    nodes = {node["id"]: node for node in graph["nodes"]}

    def chunk_of(symbol_id: str) -> str | None:
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
    """Atomically persist the graph produced by the current source set."""
    SYMBOL_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = SYMBOL_GRAPH_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_file.replace(SYMBOL_GRAPH_FILE)
    logger.info(
        "Symbol graph saved: %d nodes, %d edges.",
        len(graph["nodes"]),
        len(graph["edges"]),
    )
