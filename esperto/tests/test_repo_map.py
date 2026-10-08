"""Unit test per la mappa del repository inserita nel prompt."""

from src.core.indexing import repo_map


def _class(symbol_id: str, filename: str) -> dict:
    return {"id": symbol_id, "kind": "class", "name": symbol_id.rsplit("::", 1)[-1], "filename": filename}


def _inherits(source: str, target: str) -> dict:
    return {"source": source, "target": target, "kind": "inherits"}


GRAPH = {
    "nodes": [
        _class("m::Engine", "src/engine.py"),
        _class("m::Postgres", "src/engine.py"),
        _class("m::Sqlite", "src/engine.py"),
        _class("m::Cockroach", "src/engine.py"),
        _class("t::FakeEngine", "tests/test_engine.py"),
    ],
    "edges": [
        _inherits("m::Postgres", "m::Engine"),
        _inherits("m::Sqlite", "m::Engine"),
        _inherits("m::Cockroach", "m::Postgres"),
        _inherits("t::FakeEngine", "m::Engine"),
        {"source": "m::Postgres", "target": "m::Sqlite", "kind": "calls"},
    ],
}


def _files(tmp_path, monkeypatch, *names):
    monkeypatch.setattr(repo_map, "relative_path", lambda path: path.relative_to(tmp_path).as_posix())
    paths = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        paths.append(path)
    return paths


def test_map_has_readme_folders_and_class_hierarchy(tmp_path, monkeypatch):
    files = _files(tmp_path, monkeypatch, "README.md", "docs/README.md", "src/engine.py", "src/api.py", "tests/test_engine.py")
    files[0].write_text('<img src="logo.png" />\n\n# Progetto\nFa una cosa utile.\n', encoding="utf-8")
    files[1].write_text("# Documentazione\n", encoding="utf-8")

    text = repo_map.build_repo_map(files, GRAPH)

    # Il README e' quello piu' vicino alla radice, senza le righe iniziali di badge e HTML.
    assert text.startswith("README (README.md):\n# Progetto\nFa una cosa utile.\n\n")
    assert "logo" not in text and "Documentazione" not in text
    assert "CARTELLE:\n./ (1 file)\ndocs/ (1 file)\nsrc/ (2 file)\ntests/ (1 file)\n\n" in text
    # Le classi dei test sono escluse; prima la classe base con piu' sottoclassi.
    assert text.endswith("sottoclassi dirette):\nEngine <- Postgres, Sqlite\nPostgres <- Cockroach")


def test_folders_are_merged_and_hierarchy_cut_to_fit_the_budget(tmp_path, monkeypatch):
    files = _files(tmp_path, monkeypatch, "src/a/x.py", "src/b/y.py", "src/b/z.py")
    assert "src/a/ (1 file)\nsrc/b/ (2 file)" in repo_map.build_repo_map(files, GRAPH)

    monkeypatch.setattr(repo_map, "REPO_MAP_MAX_CHARS", 40)
    assert repo_map.build_repo_map(files, GRAPH) == "CARTELLE:\nsrc/ (3 file)"

    monkeypatch.setattr(repo_map, "REPO_MAP_MAX_CHARS", 140)
    assert repo_map.build_repo_map(files, GRAPH).endswith("Engine <- Postgres, Sqlite")
