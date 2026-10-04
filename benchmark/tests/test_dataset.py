import os
from pathlib import Path

import pytest

from src.dataset import (
    Question,
    build_zip,
    check_name,
    list_repo_files,
    load_dataset,
    validate_dataset,
)


def test_gold_set_loads_with_expected_shape():
    dataset = load_dataset("requests")
    categories = {}
    for question in dataset.questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    assert categories == {"lookup": 20, "behavior": 20, "multihop": 20, "architecture": 10, "unanswerable": 10}
    assert any(r.hop == 2 for q in dataset.questions for r in q.relevant)


@pytest.mark.skipif("REQUESTS_CHECKOUT" not in os.environ, reason="serve un checkout di requests v2.32.3")
def test_gold_set_matches_the_checkout():
    dataset = load_dataset("requests")
    assert validate_dataset(dataset, Path(os.environ["REQUESTS_CHECKOUT"])) == []


def test_question_shape_is_enforced():
    with pytest.raises(ValueError):
        Question(id="x", category="lookup", question="?")  # manca il gold
    with pytest.raises(ValueError):
        Question(id="x", category="unanswerable", question="?", key_facts=["f"])
    with pytest.raises(ValueError):
        Question(id="x", category="boh", question="?")


def test_check_name_rejects_path_traversal():
    with pytest.raises(ValueError):
        check_name("../etc")


@pytest.fixture
def tiny_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def foo():\n    pass\n\nclass Bar:\n    def baz(self):\n        pass\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("x")
    return tmp_path


def test_list_repo_files_skips_git_and_unsupported(tiny_repo):
    assert list_repo_files(tiny_repo) == ["pkg/mod.py"]


def test_build_zip_keeps_paths_at_root(tiny_repo):
    import io
    import zipfile

    content = build_zip(tiny_repo, ["pkg/mod.py"])
    assert zipfile.ZipFile(io.BytesIO(content)).namelist() == ["pkg/mod.py"]


def test_validate_dataset_flags_wrong_gold(tiny_repo):
    def question(file, symbol):
        return Question(
            id=f"{file}-{symbol}", category="lookup", question="?",
            relevant=[{"file": file, "symbol": symbol}], key_facts=["f"],
        )

    from src.dataset import Dataset

    dataset = Dataset(
        name="tiny", repo_url="u", ref="r",
        questions=[
            question("pkg/mod.py", "foo"),
            question("pkg/mod.py", "Bar"),
            question("pkg/mod.py", "baz"),  # metodo: non top-level
            question("pkg/missing.py", "foo"),
        ],
    )
    errors = validate_dataset(dataset, tiny_repo)
    assert len(errors) == 2
    assert any("baz" in e for e in errors) and any("missing.py" in e for e in errors)
