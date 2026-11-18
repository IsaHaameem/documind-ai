import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import BoundedSemaphore

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from redis import Redis
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .api import api, health
from .config import ConfigurationError, Settings, get_settings
from .database import create_database
from .documents import DocumentWorker
from .generation import DeepSeek, GenerationError
from .retrieval import ModelRuntime, RetrievalError, VectorStore

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                           "level": record.levelname, "logger": record.name, "event": record.getMessage()})


class WorkspaceBoundary:
    def __init__(self, app: ASGIApp, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/v1"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not self.settings.workspace_enabled:
            await JSONResponse({"detail": "Set SECRET_KEY to at least 32 random characters in .env."},
                               status_code=503)(scope, receive, send)
            return
        scheme, _, token = headers.get("authorization", "").partition(" ")
        expected = self.settings.secret_key.get_secret_value()
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected.encode()):
            response = JSONResponse({"detail": "Workspace access key is invalid or missing."}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        upload = scope["path"].rstrip("/") == "/api/v1/documents" and scope["method"] == "POST"
        limit = (2 * self.settings.max_upload_size_mb + 1) * 1024**2 if upload else 1024**2
        length = headers.get("content-length", "0")
        if not length.isdecimal() or len(length) > 12:
            await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
            return
        if int(length) > limit:
            await JSONResponse({"detail": "Request exceeds the server body size limit."}, status_code=413)(
                scope, receive, send
            )
            return
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Request exceeds the server body size limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        namespace = logging.getLogger("documind")
        namespace.addHandler(handler)
        namespace.setLevel("DEBUG" if settings.debug else settings.log_level)
        namespace.propagate = False
        logging.getLogger("httpx").setLevel(logging.WARNING)
        engine, sessions = create_database(settings)
        cache = Redis.from_url(settings.redis_url.get_secret_value(), socket_timeout=1, socket_connect_timeout=1)
        vectors = VectorStore(settings, ModelRuntime(settings, cache))
        provider = DeepSeek(settings)
        worker = DocumentWorker(engine, sessions, vectors, settings)
        state = application.state
        for key, value in {
            "settings": settings, "engine": engine, "sessions": sessions, "cache": cache,
            "vectors": vectors, "provider": provider, "worker": worker,
            "search_gate": BoundedSemaphore(2), "evaluation_gate": BoundedSemaphore(1),
        }.items():
            setattr(state, key, value)
        for configured, name in (
            (settings.workspace_enabled, "SECRET_KEY"), (settings.qdrant_configured, "QDRANT_URL/QDRANT_API_KEY"),
            (settings.deepseek_configured, "DEEPSEEK_API_KEY"),
        ):
            if not configured:
                logger.warning("configuration_missing name=%s", name)
        worker.thread.start()
        try:
            yield
        finally:
            worker.close()
            if not worker.thread.is_alive():
                if vectors.client is not None:
                    vectors.client.close()
                engine.dispose()
                cache.close()
            provider.client.close()
            namespace.removeHandler(handler)

    application = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
                          description="Document ingestion, retrieval, cited chat and evaluation.")
    application.add_middleware(WorkspaceBoundary, settings=settings)
    if settings.cors_origins:
        application.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        issues = [{"field": ".".join(str(item) for item in error["loc"]), "message": error["msg"]}
                  for error in exc.errors()]
        return JSONResponse({"detail": issues}, status_code=422)

    @application.exception_handler(ConfigurationError)
    async def configuration_error(_: Request, exc: ConfigurationError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @application.exception_handler(RetrievalError)
    @application.exception_handler(GenerationError)
    async def provider_error(_: Request, exc: Exception):
        logger.warning("provider_failure type=%s", type(exc).__name__)
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @application.exception_handler(SQLAlchemyError)
    async def database_error(_: Request, exc: SQLAlchemyError):
        logger.error("database_failure type=%s", type(exc).__name__)
        return JSONResponse({"detail": "Docker PostgreSQL operation failed. Check backend and database health."},
                            status_code=503)

    @application.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception):
        logger.error("unhandled_request_failure type=%s", type(exc).__name__)
        return JSONResponse({"detail": "Unexpected backend error."}, status_code=500)

    application.include_router(health)
    application.include_router(api)
    return application


app = create_app()
