import math
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from documind.config import Settings
from documind.ingestion import Chunk
from documind.retrieval import Hit, ModelRuntime, RetrievalError, VectorStore


class MemoryCache:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value


class ThreeDimensionalRuntime(ModelRuntime):
    dimension = 3

    def encode(self, texts):
        return [[1.0, 0, 0] if "refund" in text.casefold() else [0, 1.0, 0] for text in texts]


@pytest.fixture
def store():
    settings = Settings(_env_file=None)
    result = VectorStore(settings, ThreeDimensionalRuntime(settings, MemoryCache()))
    result.client = QdrantClient(":memory:")
    yield result
    result.client.close()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
def test_dense_sparse_hybrid_scope_and_delete(store):
    refunds, logs = uuid4(), uuid4()
    refund_chunk = Chunk(uuid4(), "RET-042: request a refund within 30 days.", "Page 4", 4)
    log_chunk = Chunk(uuid4(), "SEC-220: access logs are kept for 90 days.", "Page 2", 2)
    store.index(refunds, "returns.pdf", [refund_chunk])
    store.index(logs, "privacy.pdf", [log_chunk])
    dense, cache_hit = store.query("refund window", [refunds, logs], "dense", 3, 0.3)
    assert not cache_hit
    assert [hit.chunk_id for hit in dense] == [refund_chunk.id]
    assert store.query("refund window", [refunds], "dense", 3, 0.3)[1]
    for mode in ("sparse", "hybrid"):
        results, _ = store.query("SEC-220", [refunds, logs], mode, 3, 0.3)
        assert results[0].chunk_id == log_chunk.id
    scoped, _ = store.query("SEC-220", [refunds], "sparse", 3, 0.3)
    assert scoped == []
    assert store.query("SEC-220", [], "hybrid", 3, 0.3) == ([], False)
    store.delete(logs)
    assert store.query("SEC-220", [logs], "sparse", 3, 0.3)[0] == []
    assert store.client.count(store.name, exact=True).count == 1


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
def test_incompatible_embeddings_rejected(store):
    store.check_collection(create=True)
    store.runtime.settings = Settings(_env_file=None, embedding_revision="a" * 40)
    with pytest.raises(RetrievalError, match="differs"):
        store.check_collection()
