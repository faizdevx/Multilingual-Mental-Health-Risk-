import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.app.reports import Reports
from src.inference import ModelRegistry

MARKER = "ZXQ-PRIVATE-MARKER-4417 jane.doe@example.com"


@pytest.fixture
def client(registry, tmp_path):
    return TestClient(create_app(registry, Reports(metrics_dir=tmp_path / "m", reports_dir=tmp_path / "r", processed_dir=tmp_path / "p", ckpt_dir=tmp_path / "c")))


def test_root_json_and_html(client):
    j = client.get("/", headers={"accept": "application/json"})
    assert j.status_code == 200 and "not a medical diagnostic tool" in j.json()["disclaimer"]
    h = client.get("/")
    assert h.status_code == 200 and "Dashboard" in h.text and "Not available" in h.text  # no metrics -> honest placeholders


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "healthy" and set(j["models_loaded"]) == {"bilstm", "transformer"} and j["device"] == "cpu"


def test_predict_ok_and_validation(client):
    r = client.post("/predict", json={"text": "the weather is nice today", "model": "transformer"})
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "transformer" and 0 <= b["risk_probability"] <= 1 and b["inference_latency_ms"] >= 0
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 6000}).status_code == 422
    assert client.post("/predict", json={"text": "hello", "model": "nope"}).status_code == 400


def test_validation_errors_do_not_echo_input(client):
    r = client.post("/predict", json={"text": MARKER * 400})  # too long -> 422
    assert r.status_code == 422 and "ZXQ-PRIVATE" not in r.text


@pytest.mark.parametrize("path", ["/", "/analyze", "/playground", "/models", "/languages", "/errors", "/batch", "/health-ui", "/about"])
def test_ui_routes(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "not a medical diagnostic tool" in r.text.lower()  # disclaimer banner on every page


def test_ui_pages_show_not_available_without_artifacts(client):
    assert "Not available" in client.get("/models").text and "Not available" in client.get("/errors").text
    assert client.get("/languages").status_code == 200


def test_compare_endpoint(client):
    r = client.post("/api/compare", json={"text": "the weather is nice"})
    assert r.status_code == 200 and set(r.json()["results"]) == {"bilstm", "transformer"}


def test_json_api_views(client):
    for p in ("/api/dashboard", "/api/models", "/api/languages", "/api/errors"):
        assert client.get(p).status_code == 200


def test_batch_json_and_csv_and_no_disk_writes(client, tmp_path, monkeypatch):
    import os
    before = set(os.listdir(os.getcwd()))
    csv_bytes = "id,text,language\n1,the weather is nice,\n=cmd|x,yaar aaj mood off hai,hi-Latn\n3,,\n".encode()
    r = client.post("/api/batch", files={"file": ("in.csv", io.BytesIO(csv_bytes), "text/csv")}, data={"model": "bilstm"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3 and rows[1]["language"] == "hi-Latn" and rows[1]["id"].startswith("'=")  # formula injection neutralised
    assert rows[2]["note"] == "empty text" and 0 <= rows[0]["risk_probability"] <= 1
    assert set(rows[0]) >= {"id", "language", "sentiment", "emotion", "risk", "risk_probability"}
    c = client.post("/api/batch", files={"file": ("in.csv", io.BytesIO(csv_bytes), "text/csv")}, data={"model": "bilstm", "format": "csv"})
    assert c.headers["content-type"].startswith("text/csv") and c.text.splitlines()[0].startswith("id,language")
    assert set(os.listdir(os.getcwd())) == before  # nothing persisted


def test_batch_rejects_bad_input(client):
    bad = client.post("/api/batch", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 400
    big = "text\n" + "hello\n" * 600
    assert client.post("/api/batch", files={"file": ("x.csv", io.BytesIO(big.encode()), "text/csv")}).status_code == 413


def test_no_model_loaded_returns_503(tmp_path):
    c = TestClient(create_app(ModelRegistry(adapters={}), Reports(metrics_dir=tmp_path)))
    assert c.post("/predict", json={"text": "hello"}).status_code == 503
    assert c.get("/health").json()["models_loaded"] == []


def test_submitted_text_never_logged(client, caplog, capsys):
    with caplog.at_level(logging.DEBUG, logger="mhrisk"):
        client.post("/predict", json={"text": MARKER, "model": "bilstm"})
        client.post("/predict", json={"text": MARKER, "model": "bilstm", "explain": True})
        client.post("/api/batch", files={"file": ("in.csv", io.BytesIO(f"text\n{MARKER}\n".encode()), "text/csv")}, data={"model": "bilstm"})
    out = capsys.readouterr()
    logged = caplog.text + out.out + out.err
    assert "ZXQ-PRIVATE" not in logged and "example.com" not in logged
    assert "request_id" in logged  # metadata IS logged


def test_log_event_drops_unsafe_fields():
    from src.privacy import log_event
    rec = log_event(request_id="abc", text="secret words", email="a@b.c", latency_ms=1.0)
    assert "text" not in rec and "email" not in rec and rec["request_id"] == "abc" and rec["dropped_fields"] == 2


def test_responses_not_cached_and_request_id(client):
    r = client.get("/health")
    assert r.headers["cache-control"] == "no-store" and len(r.headers["x-request-id"]) == 16
