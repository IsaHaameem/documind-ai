import hashlib
import json
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal
from uuid import UUID
from zlib import crc32

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from redis import Redis
from redis.exceptions import RedisError

from .config import ConfigurationError, Settings
from .ingestion import Chunk

logger = logging.getLogger(__name__)
SearchMode = Literal["dense", "sparse", "hybrid"]
TOKEN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", re.UNICODE)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SPARSE_SCHEMA = "lexical-bm25-crc32-v1"


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    location: str
    page: int | None
    text: str
    retrieval_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    candidates: list[Hit]
    hits: list[Hit]
    latency_ms: dict[str, float]
    embedding_cache_hit: bool


def sparse_vector(text: str, *, query: bool) -> models.SparseVector:
    tokens = TOKEN.findall(text.casefold())
    counts = Counter(crc32(token.encode("utf-8")) for token in tokens)
    length_normalizer = 1.2 * (0.25 + 0.75 * len(tokens) / 100)
    indices = sorted(counts)
    values = [1.0 if query else 2.2 * counts[index] / (counts[index] + length_normalizer) for index in indices]
    return models.SparseVector(indices=indices, values=values)


class ModelRuntime:
    def __init__(self, settings: Settings, cache: Redis):
        self.settings = settings
        self.cache = cache
        self.encoder = None
        self.cross_encoder = None
        self.lock = threading.RLock()

    @property
    def signature(self) -> dict[str, str]:
        return {"embedding_model": self.settings.embedding_model,
                "embedding_revision": self.settings.embedding_commit,
                "dense_schema": "normalized-bge-query-prefix-v1", "sparse_schema": SPARSE_SCHEMA}

    def _encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                self.settings.embedding_model, revision=self.settings.embedding_commit,
                device="cpu", trust_remote_code=False, model_kwargs={"use_safetensors": True},
            )
        return self.encoder

    @property
    def dimension(self) -> int:
        with self.lock:
            dimension = self._encoder().get_sentence_embedding_dimension()
        if not dimension:
            raise RetrievalError("Embedding model does not declare a vector dimension.")
        return dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            with self.lock:
                vectors = self._encoder().encode(
                    texts, batch_size=self.settings.embedding_batch_size,
                    normalize_embeddings=True, show_progress_bar=False,
                ).tolist()
                dimension = self.dimension
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("Embedding model could not run. Check model availability and backend logs.") from None
        if len(vectors) != len(texts) or any(
            len(vector) != dimension or any(not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise RetrievalError("Embedding model returned invalid vectors.")
        return vectors

    def query_vector(self, query: str) -> tuple[list[float], bool]:
        ttl = self.settings.embedding_cache_ttl_seconds
        digest = hashlib.sha256(json.dumps([self.signature, query], sort_keys=True).encode()).hexdigest()
        key = f"documind:embedding:{digest}"
        if ttl:
            try:
                cached = self.cache.get(key)
                if cached is not None:
                    vector = json.loads(cached)
                    if isinstance(vector, list) and len(vector) == self.dimension and all(
                        isinstance(value, (float, int)) and math.isfinite(value) for value in vector
                    ):
                        return vector, True
            except (RedisError, ValueError, TypeError):
                logger.warning("embedding_cache_read_failed")
        vector = self.encode([BGE_QUERY_PREFIX + query])[0]
        if ttl:
            try:
                self.cache.setex(key, ttl, json.dumps(vector, separators=(",", ":")))
            except RedisError:
                logger.warning("embedding_cache_write_failed")
        return vector, False

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        try:
            with self.lock:
                if self.cross_encoder is None:
                    from sentence_transformers import CrossEncoder

                    self.cross_encoder = CrossEncoder(
                        self.settings.reranker_model, revision=self.settings.reranker_commit, device="cpu",
                        max_length=512, trust_remote_code=False, model_kwargs={"use_safetensors": True},
                    )
                scores = self.cross_encoder.predict(
                    [(query, hit.text) for hit in hits], batch_size=min(16, self.settings.embedding_batch_size),
                    show_progress_bar=False,
                )
            ranked = [replace(hit, rerank_score=float(score)) for hit, score in zip(hits, scores, strict=True)]
            if any(not math.isfinite(hit.rerank_score) for hit in ranked):
                raise ValueError("Non-finite reranker score.")
        except Exception:
            raise RetrievalError("Reranker could not run. Check RERANKER_MODEL or disable reranking.") from None
        return sorted(ranked, key=lambda hit: hit.rerank_score, reverse=True)


class VectorStore:
    def __init__(self, settings: Settings, runtime: ModelRuntime):
        self.name = settings.qdrant_collection_name
        self.settings = settings
        self.runtime = runtime
        self.lock = threading.Lock()
        self.client = (QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key.get_secret_value(),
                                   timeout=12, check_compatibility=False)
                       if settings.qdrant_configured else None)

    def require_client(self) -> QdrantClient:
        if self.client is None:
            raise ConfigurationError("Set hosted QDRANT_URL and QDRANT_API_KEY in .env.")
        return self.client

    def check_collection(self, *, create: bool = False, dense: bool = True) -> bool:
        client = self.require_client()
        with self.lock:
            try:
                if not client.collection_exists(self.name):
                    if not create:
                        return False
                    try:
                        client.create_collection(
                            self.name,
                            vectors_config={"dense": models.VectorParams(size=self.runtime.dimension,
                                                                         distance=models.Distance.COSINE)},
                            sparse_vectors_config={"lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                            metadata=self.runtime.signature,
                        )
                    except UnexpectedResponse:
                        if not client.collection_exists(self.name):
                            raise
                collection = client.get_collection(self.name)
                params = collection.config
                vectors = params.params.vectors
                sparse = (params.params.sparse_vectors or {}).get("lexical")
                if (params.metadata != self.runtime.signature or not isinstance(vectors, dict) or "dense" not in vectors
                        or vectors["dense"].distance != models.Distance.COSINE or sparse is None
                        or sparse.modifier != models.Modifier.IDF
                        or (dense and vectors["dense"].size != self.runtime.dimension)):
                    raise RetrievalError("Qdrant collection schema or embedding model differs. Choose a fresh collection and reindex.")
                if create and "document_id" not in collection.payload_schema:
                    client.create_payload_index(self.name, "document_id", models.PayloadSchemaType.KEYWORD, wait=True)
                return True
            except RetrievalError:
                raise
            except Exception:
                raise RetrievalError("Hosted Qdrant is unavailable or the API key lacks collection permissions.") from None

    @staticmethod
    def filter_for(ids: list[UUID]) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchAny(any=[str(identifier) for identifier in ids])
        )])

    def index(self, document_id: UUID, filename: str, chunks: list[Chunk]) -> None:
        client = self.require_client()
        self.check_collection(create=True)
        # Remove any partial write left by a failed or interrupted ingestion.
        self.delete(document_id)
        for offset in range(0, len(chunks), 32):
            batch = chunks[offset:offset + 32]
            vectors = self.runtime.encode([chunk.text for chunk in batch])
            points = [models.PointStruct(
                id=str(chunk.id),
                vector={"dense": vector, "lexical": sparse_vector(chunk.text, query=False)},
                payload={"document_id": str(document_id), "filename": filename, "location": chunk.location,
                         "page": chunk.page, "text": chunk.text},
            ) for chunk, vector in zip(batch, vectors, strict=True)]
            try:
                client.upsert(self.name, points, wait=True)
            except Exception:
                raise RetrievalError("Could not index passages in hosted Qdrant. Check Qdrant connectivity and write permissions.") from None

    def delete(self, document_id: UUID) -> None:
        client = self.require_client()
        try:
            if client.collection_exists(self.name):
                client.delete(self.name, models.FilterSelector(filter=self.filter_for([document_id])), wait=True)
        except Exception:
            raise RetrievalError("Could not remove the document vectors from hosted Qdrant.") from None

    def query(self, question: str, ids: list[UUID], mode: SearchMode, limit: int,
              dense_score: float) -> tuple[list[Hit], bool]:
        if not ids:
            return [], False
        client = self.require_client()
        if not self.check_collection(dense=mode != "sparse"):
            raise RetrievalError("Qdrant collection is missing. Restore the collection or reindex the affected documents.")
        filters = self.filter_for(ids)
        dense, cache_hit = self.runtime.query_vector(question) if mode != "sparse" else (None, False)
        sparse = sparse_vector(question, query=True) if mode != "dense" else None
        if sparse is not None and not sparse.indices:
            if mode == "sparse":
                return [], False
            mode = "dense"
        try:
            if mode == "hybrid":
                depth = min(100, limit * 3)
                response = client.query_points(
                    self.name, prefetch=[
                        models.Prefetch(query=dense, using="dense", filter=filters, limit=depth, score_threshold=dense_score),
                        models.Prefetch(query=sparse, using="lexical", filter=filters, limit=depth),
                    ], query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=filters,
                    limit=limit, with_payload=True,
                )
            else:
                response = client.query_points(
                    self.name, query=dense if mode == "dense" else sparse,
                    using="dense" if mode == "dense" else "lexical", query_filter=filters,
                    score_threshold=dense_score if mode == "dense" else None, limit=limit, with_payload=True,
                )
            return [Hit(UUID(str(point.id)), UUID(point.payload["document_id"]), point.payload["filename"],
                        point.payload["location"], point.payload.get("page"), point.payload["text"], point.score)
                    for point in response.points], cache_hit
        except Exception:
            raise RetrievalError("Hosted Qdrant query failed.") from None

    def search(self, question: str, ids: list[UUID], mode: SearchMode, limit: int,
               dense_score: float, rerank: bool) -> SearchResult:
        started = perf_counter()
        candidates, cached = self.query(question, ids, mode, limit, dense_score)
        retrieved = perf_counter()
        hits = (self.runtime.rerank(question, candidates)[:self.settings.rerank_top_k]
                if rerank else candidates)
        finished = perf_counter()
        return SearchResult(candidates, hits, {
            "retrieval": round((retrieved - started) * 1000, 2),
            "reranking": round((finished - retrieved) * 1000, 2),
            "total": round((finished - started) * 1000, 2),
        }, cached)
