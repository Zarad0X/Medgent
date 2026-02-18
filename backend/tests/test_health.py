from fastapi.testclient import TestClient

from app.main import app

AUTH_HEADERS = {"X-API-Key": "dev-local-key"}


def test_health_endpoints():
    with TestClient(app) as client:
        live = client.get("/api/v1/health/live")
        assert live.status_code == 200
        assert live.json()["status"] == "ok"

        ready = client.get("/api/v1/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_protected_endpoint_requires_api_key():
    with TestClient(app) as client:
        resp = client.post("/api/v1/cases", json={"patient_pseudo_id": "p-unauth"})
        assert resp.status_code == 401


def test_inference_ping_mock_provider():
    with TestClient(app) as client:
        resp = client.get("/api/v1/inference/ping", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "mock"
        assert body["reachable"] is True
