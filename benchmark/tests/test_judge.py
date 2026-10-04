import json

from src import judge


def _chat_returning(payload):
    return lambda *args, **kwargs: payload if isinstance(payload, str) else json.dumps(payload)


def test_judge_facts_returns_one_verdict_per_fact():
    chat = _chat_returning({"verdicts": [True, False]})
    assert judge.judge_facts("q", "a", ["f1", "f2"], chat=chat) == [True, False]


def test_judge_facts_rejects_wrong_length_or_invalid_json():
    assert judge.judge_facts("q", "a", ["f1", "f2"], chat=_chat_returning({"verdicts": [True]})) is None
    assert judge.judge_facts("q", "a", ["f1"], chat=_chat_returning("not json")) is None
    assert judge.judge_facts("q", "a", ["f1"], chat=_chat_returning("[1, 2]")) is None


def test_boolean_judges():
    assert judge.judge_faithful("q", "a", "ctx", chat=_chat_returning({"value": True})) is True
    assert judge.judge_abstains("q", "a", chat=_chat_returning({"value": False})) is False
    assert judge.judge_abstains("q", "a", chat=_chat_returning({"other": 1})) is None


def test_faithful_prompt_truncates_the_context(monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_CONTEXT_MAX_CHARS", 10)
    seen = {}

    def chat(prompt, **kwargs):
        seen["prompt"] = prompt
        return json.dumps({"value": True})

    judge.judge_faithful("q", "a", "x" * 100, chat=chat)
    assert "x" * 10 in seen["prompt"] and "x" * 11 not in seen["prompt"]
