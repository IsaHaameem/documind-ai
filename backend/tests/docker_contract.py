"""Exercise the full API on isolated Docker PostgreSQL and Redis.

Run only with APP_ENV=test and no hosted credentials. Qdrant is embedded
in memory for this test. DeepSeek is substituted with a deterministic reply.
"""

import json
import os
import warnings
from io import BytesIO
from time import monotonic, sleep
from unittest.mock import patch
from uuid import uuid4

import pymupdf
from docx import Document
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from documind.config import Settings, get_settings
from documind.main import create_app
from documind.retrieval import ModelRuntime, VectorStore


class TestSettings(Settings):
    @property
    def qdrant_configured(self):
        return True

    @property
    def deepseek_configured(self):
        return True


class ContractEncoder(ModelRuntime):
    dimension = 3

    @property
    def signature(self):
        return {"testing": "contract-encoder-v1"}

    def encode(self, texts):
        return [[1.0, 0, 0] if "refund" in text.lower() else [0, 1.0, 0] for text in texts]

    def rerank(self, question, hits):
        return hits


def local_store(settings, runtime):
    # Construct the store without a hosted client and inject a test-only index.
    encoder = runtime if os.environ.get("DOCUMIND_CONTRACT_REAL_MODELS") == "true" else ContractEncoder(
        settings, runtime.cache
    )
    store = VectorStore(Settings(_env_file=None), encoder)
    store.client = QdrantClient(":memory:")
    return store


def grounded_reply(system, payload, *, max_tokens=1300):
    if "previous_questions" in payload:
        return json.dumps({"query": "What is the refund deadline?"})
    for source in payload["sources"]:
        if "30 calendar days" in source["excerpt"]:
            return json.dumps({"insufficient_context": False, "claims": [
                {"statement": "The deadline is 30 calendar days.", "evidence": [
                    {"source": source["source"], "quote": "30 calendar days"}
                ]}
            ]})
    return json.dumps({"insufficient_context": True, "claims": []})


def wait_until(fetch, condition):
    end = monotonic() + 25
    while monotonic() < end:
        result = fetch()
        if condition(result):
            return result
        sleep(0.15)
    raise AssertionError("Timed out waiting for worker job.")


def assert_ok(response, status):
    assert response.status_code == status, (response.status_code, response.text)
    return response.json()


def run():
    original = get_settings()
    if original.app_env != "test" or original.qdrant_configured or original.deepseek_configured:
        raise SystemExit("Use an isolated APP_ENV=test stack with blank hosted credentials.")
    settings = TestSettings(_env_file=None, **original.model_dump())
    suffix = str(uuid4())[:8]
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page().insert_text((40, 40), f"Cust-{suffix}: refund requests are accepted within 30 calendar days.")
        pdf_data = pdf.tobytes()
    document = Document()
    document.add_paragraph(f"SEC-{suffix}: access logs are retained for seven days.")
    buffer = BytesIO()
    document.save(buffer)
    headers = {"Authorization": "Bearer " + original.secret_key.get_secret_value()}
    with warnings.catch_warnings(), patch("documind.main.VectorStore", side_effect=local_store):
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        app = create_app(settings)
        with TestClient(app, headers=headers) as client:
            app.state.provider.json_response = grounded_reply
            assert_ok(client.get("/health/ready"), 200)
            response = assert_ok(client.post("/api/v1/documents", files=[
                ("files", ("returns.pdf", pdf_data, "application/pdf")),
                ("files", ("security.docx", buffer.getvalue(), "application/octet-stream")),
                ("files", ("notes.txt", f"Warehouse notes {suffix}".encode(), "text/plain")),
                ("files", ("broken.txt", b"  ", "text/plain")),
                ("files", ("bad.exe", b"unrecognized", "application/octet-stream")),
            ]), 202)
            assert response[4]["error"]
            ids = [item["document"]["id"] for item in response[:4]]
            try:
                for identifier, expected in zip(ids, ("ready", "ready", "ready", "failed"), strict=True):
                    fetch_status = lambda: assert_ok(client.get(f"/api/v1/documents/{identifier}"), 200)
                    row = wait_until(fetch_status, lambda row: row["status"] in ("ready", "failed"))
                    assert row["status"] == expected, row
                duplicate = assert_ok(client.post("/api/v1/documents", files=[
                    ("files", ("copy.pdf", pdf_data, "application/pdf"))
                ]), 202)
                assert duplicate[0]["duplicate"] and duplicate[0]["document"]["id"] == ids[0]

                for mode in ("dense", "sparse", "hybrid"):
                    result = assert_ok(client.post("/api/v1/retrieval", json={
                        "question": "refund deadline", "document_ids": [ids[0]], "mode": mode, "rerank": True,
                    }), 200)
                    assert result["hits"][0]["page"] == 2
                result = assert_ok(client.post("/api/v1/retrieval", json={
                    "question": "refund deadline", "document_ids": [ids[1]], "mode": "sparse", "rerank": False,
                }), 200)
                assert result["hits"] == []
                assert_ok(client.post("/api/v1/retrieval", json={
                    "question": "refund deadline", "document_ids": [ids[0]], "mode": "dense", "rerank": False,
                }), 200)["embedding_cache_hit"]
                assert app.state.cache.ping()

                first = assert_ok(client.post("/api/v1/chat", json={
                    "question": "What is the refund deadline?", "document_ids": [ids[0]],
                }), 200)
                conversation = first["conversation_id"]
                assert first["message"]["citations"][0]["document_id"] == ids[0]
                follow = assert_ok(client.post("/api/v1/chat", json={
                    "question": "Can you repeat the deadline?", "conversation_id": conversation, "document_ids": [ids[0]],
                }), 200)
                assert follow["search_query"] == "What is the refund deadline?"
                history = assert_ok(client.get(f"/api/v1/conversations/{conversation}"), 200)
                assert [message["role"] for message in history["messages"]] == ["user", "assistant"] * 2
                assert client.delete(f"/api/v1/conversations/{conversation}").status_code == 204
                assert client.get(f"/api/v1/conversations/{conversation}").status_code == 404

                evaluation = assert_ok(client.post("/api/v1/evaluations", json={
                    "top_k": 2, "document_ids": [ids[0]], "cases": [{
                        "question": "What is the refund deadline?",
                        "expected_answer": "30 calendar days",
                        "relevant_passages": ["refund requests are accepted within 30 calendar days."],
                    }]
                }), 201)
                assert set(evaluation["report"]["modes"]) == {"dense", "sparse", "hybrid", "hybrid_rerank"}
                assert evaluation["id"] == assert_ok(client.get("/api/v1/evaluations?limit=1"), 200)[0]["id"]
                assert_ok(client.delete(f"/api/v1/documents/{ids[0]}"), 202)
                wait_until(lambda: client.get(f"/api/v1/documents/{ids[0]}"), lambda result: result.status_code == 404)
                assert app.state.vectors.client.count(
                    app.state.vectors.name, count_filter=app.state.vectors.filter_for([ids[0]]), exact=True,
                ).count == 0
            finally:
                for identifier in ids:
                    client.delete(f"/api/v1/documents/{identifier}")
                for identifier in ids:
                    wait_until(lambda: client.get(f"/api/v1/documents/{identifier}"), lambda result: result.status_code == 404)
    print("Docker contract passed: uploads, jobs, deduplication, retrieval, Redis, citations, chat and vector deletion.")


if __name__ == "__main__":
    run()
