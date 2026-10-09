from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_or_malicious_run_id_is_404():
    assert client.get("/runs/does-not-exist").status_code == 404
    assert client.get("/runs/..%2Fetc").status_code == 404
    assert client.get("/runs/does-not-exist/compare/neither").status_code == 404


def test_start_run_validates_the_spec():
    assert client.post("/runs", json={"dataset": "../x"}).status_code == 400
    assert client.post("/runs", json={"dataset": "requests", "retrieval_configs": ["nope"]}).status_code == 400


def test_second_run_is_rejected_while_one_is_active(monkeypatch):
    monkeypatch.setattr(api, "_active_run", "busy-run")
    response = client.post("/runs", json={"dataset": "requests"})
    assert response.status_code == 409
