import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from documind.config import ConfigurationError, Settings
from documind.main import create_app

TOKEN = "local-test-workspace-access-only-123456"


@pytest.fixture
def client():
    settings = Settings(_env_file=None, postgres_password="local-test-db-only", secret_key=TOKEN,
                        max_upload_size_mb=1, cors_origins=["https://example.test"])
    with TestClient(create_app(settings)) as session:
        yield session


def test_configuration_rejects_host_database_and_invalid_chunks():
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://u:p@localhost:5432/test")
    with pytest.raises(ConfigurationError, match="Docker postgres"):
        settings.db_url()
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_public_health_and_protected_routes(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/api/v1/dashboard").status_code == 401
    openapi = client.get("/openapi.json").json()
    assert openapi["paths"]["/api/v1/dashboard"]["get"]["security"] == [{"HTTPBearer": []}]


def test_invalid_requests_and_missing_qdrant_return_clear_errors(client):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    invalid = client.post("/api/v1/retrieval", headers=headers, json={"question": "  ", "top_k": 500})
    assert invalid.status_code == 422
    assert all("input" not in issue for issue in invalid.json()["detail"])
    valid = client.post("/api/v1/retrieval", headers=headers, json={"question": "What is the refund window?"})
    assert valid.status_code == 503
    assert "QDRANT_URL" in valid.json()["detail"]


def test_size_limit_and_cors(client):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Length": str(3 * 1024**2 + 1)}
    assert client.post("/api/v1/documents", headers=headers).status_code == 413
    response = client.options("/api/v1/dashboard", headers={
        "Origin": "https://example.test", "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.test"
