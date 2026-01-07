from fastapi.testclient import TestClient
from rag_tools.api.main import app

client = TestClient(app)

def test_llm_metrics_baseline():
    payload = {"reference": "Добавить пациента", "candidate": "Добавить пациента"}
    r = client.post("/metrics/llm", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    assert "exact_match" in data["metrics"]
    assert "token_f1" in data["metrics"]

def test_rag_metrics_baseline():
    payload = {
        "relevant": [[3], [10, 11]],
        "predicted": [[1, 2, 3], [11, 9, 10]],
        "k": 3
    }
    r = client.post("/metrics/rag", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["k"] == 3
    assert "metrics" in data
    assert "hit@3" in data["metrics"]
    assert "mrr" in data["metrics"]