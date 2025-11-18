"""Basic health endpoint test."""
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ping() -> None:
    response = client.get("/health/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
