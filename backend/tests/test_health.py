from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_liveness():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "pozz"}


def test_health_ready_degraded_without_database_url():
    response = client.get("/api/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"] == "error"
