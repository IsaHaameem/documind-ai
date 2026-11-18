import logging
from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.security import HTTPBearer
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from .config import ConfigurationError
from .conversations import ConversationChanged, ConversationMissing, previous_questions, save_turn
from .database import Conversation, Document, EvaluationRun, Message
from .documents import Upload, document_summary, enqueue, ready_scope
from .evaluation import demo_bytes, demo_cases, demo_digest, evaluate
from .generation import generate, rewrite
from .ingestion import DocumentError, check_file_header, upload_filename
from .retrieval import RetrievalError
from .schemas import (ChatInput, ChatView, ConversationHistory, ConversationInput, ConversationView,
                      Dashboard, DocumentView, EvaluationInput, EvaluationView, RetrievalView,
                      SearchInput, UploadResult)

logger = logging.getLogger(__name__)
health = APIRouter(tags=["Health"])
api = APIRouter(prefix="/api/v1", dependencies=[Depends(HTTPBearer(auto_error=False))])
PageSize = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


@contextmanager
def limited(semaphore: BoundedSemaphore):
    if not semaphore.acquire(blocking=False):
        raise HTTPException(429, "The server is busy. Try again shortly.", headers={"Retry-After": "5"})
    try:
        yield
    finally:
        semaphore.release()


@health.get("/health/live")
def live():
    return {"status": "ok"}


@health.get("/health/ready")
def ready(request: Request):
    try:
        with request.app.state.sessions() as session:
            session.execute(text("SELECT 1 FROM alembic_version LIMIT 1"))
    except SQLAlchemyError:
        raise HTTPException(503, "Docker PostgreSQL is unavailable or migrations have not run.") from None
    return {"status": "ok"}


@api.get("/settings", tags=["System"])
def settings_status(request: Request, check_external: bool = False):
    state = request.app.state
    settings = state.settings
    qdrant = "configured" if settings.qdrant_configured else "Set hosted QDRANT_URL and QDRANT_API_KEY in .env."
    deepseek = "configured" if settings.deepseek_configured else "Set DEEPSEEK_API_KEY in .env."
    if check_external and settings.qdrant_configured:
        try:
            exists = state.vectors.check_collection(dense=False)
            qdrant = "collection ready" if exists else "reachable; collection will be created on first upload"
        except RetrievalError as exc:
            qdrant = str(exc)
    if check_external and settings.deepseek_configured:
        deepseek = state.provider.check_models()
    try:
        cache = "reachable" if state.cache.ping() else "unavailable"
    except Exception:
        cache = "unavailable; embeddings will be computed without cache"
    return {"application": settings.app_name, "environment": settings.app_env, "qdrant": qdrant,
            "deepseek": deepseek, "redis": cache, "embedding_model": settings.embedding_model,
            "reranker": settings.reranker_model if settings.rerank_enabled else "disabled",
            "answer_model": settings.deepseek_model, "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "max_upload_size_mb": settings.max_upload_size_mb,
            "max_upload_files": settings.max_upload_files, "rerank_top_k": settings.rerank_top_k}


@api.get("/dashboard", response_model=Dashboard, tags=["Documents"])
def dashboard(request: Request):
    with request.app.state.sessions() as session:
        data = document_summary(session)
        recent = session.scalars(select(Document).order_by(Document.updated_at.desc(), Document.id).limit(8)).all()
        return Dashboard(**data, recent=[DocumentView.model_validate(row) for row in recent])


@api.get("/documents", response_model=list[DocumentView], tags=["Documents"])
def list_documents(request: Request, limit: PageSize = 50, offset: Offset = 0, ready_only: bool = False):
    with request.app.state.sessions() as session:
        statement = select(Document).order_by(Document.created_at.desc(), Document.id).offset(offset).limit(limit)
        if ready_only:
            statement = statement.where(Document.status == "ready")
        return [DocumentView.model_validate(row) for row in session.scalars(statement)]


@api.get("/documents/{identifier}", response_model=DocumentView, tags=["Documents"])
def document_status(request: Request, identifier: UUID):
    with request.app.state.sessions() as session:
        document = session.get(Document, identifier)
        if document is None:
            raise HTTPException(404, "Document not found.")
        return DocumentView.model_validate(document)


@api.post("/documents", response_model=list[UploadResult], status_code=202, tags=["Documents"])
def create_documents(request: Request, files: Annotated[list[UploadFile], File()]):
    state = request.app.state
    state.vectors.require_client()
    settings = state.settings
    if not 1 <= len(files) <= settings.max_upload_files:
        raise HTTPException(422, f"Upload between 1 and {settings.max_upload_files} files per request.")
    results = []
    max_bytes = settings.max_upload_size_mb * 1024**2
    for file in files:
        try:
            filename, extension = upload_filename(file.filename)
            content = file.file.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise DocumentError(f"File exceeds the {settings.max_upload_size_mb} MiB limit.")
            check_file_header(content, extension)
            with state.sessions.begin() as session:
                row, duplicate = enqueue(session, Upload(filename, extension, content))
                results.append(UploadResult(filename=filename, document=DocumentView.model_validate(row),
                                            duplicate=duplicate))
        except DocumentError as exc:
            results.append(UploadResult(filename=(file.filename or "Unnamed file")[:255], error=str(exc)))
    state.worker.wakeup.set()
    if not any(result.document for result in results):
        raise HTTPException(422, [result.model_dump(exclude_none=True) for result in results])
    return results


@api.post("/documents/{identifier}/retry", response_model=DocumentView, status_code=202, tags=["Documents"])
def retry_document(request: Request, identifier: UUID):
    state = request.app.state
    state.vectors.require_client()
    with state.sessions.begin() as session:
        row = session.get(Document, identifier, with_for_update=True)
        if row is None:
            raise HTTPException(404, "Document not found.")
        if row.status != "failed" or row.content is None:
            raise HTTPException(409, "Only failed ingestion jobs with retained uploads can be retried.")
        row.status = "queued"
        row.error = None
        session.flush()
        view = DocumentView.model_validate(row)
    state.worker.wakeup.set()
    return view


@api.delete("/documents/{identifier}", status_code=202, response_model=DocumentView, tags=["Documents"])
def delete_document(request: Request, identifier: UUID):
    state = request.app.state
    state.vectors.require_client()
    with state.sessions.begin() as session:
        row = session.get(Document, identifier, with_for_update=True)
        if row is None:
            raise HTTPException(404, "Document not found.")
        row.status = "deleting"
        row.error = None
        session.flush()
        view = DocumentView.model_validate(row)
    state.worker.wakeup.set()
    return view


def run_search(state, question: str, data: SearchInput, scope: list[UUID]):
    settings = state.settings
    if data.rerank is True and not settings.rerank_enabled:
        raise HTTPException(422, "Reranking is disabled by the server configuration.")
    rerank = settings.rerank_enabled if data.rerank is None else data.rerank
    return state.vectors.search(
        question, scope, data.mode, data.top_k or settings.retrieval_top_k,
        settings.dense_min_score if data.dense_min_score is None else data.dense_min_score, rerank,
    )


@api.post("/retrieval", response_model=RetrievalView, tags=["Retrieval"])
def retrieval(request: Request, data: SearchInput):
    state = request.app.state
    state.vectors.require_client()
    with limited(state.search_gate):
        with state.sessions() as session:
            scope = ready_scope(session, data.document_ids)
        return run_search(state, data.question, data, scope)


@api.post("/chat", response_model=ChatView, tags=["Chat"])
def chat(request: Request, data: ChatInput):
    state = request.app.state
    state.vectors.require_client()
    if not state.settings.deepseek_configured:
        raise ConfigurationError("Set DEEPSEEK_API_KEY in .env to enable chat.")
    with limited(state.search_gate):
        earlier, previous = [], None
        try:
            with state.sessions() as session:
                scope = ready_scope(session, data.document_ids)
                if data.conversation_id:
                    earlier, previous = previous_questions(session, data.conversation_id)
            query = rewrite(state.provider, data.question, earlier) if scope else data.question
            result = run_search(state, query, data, scope)
            answer = generate(state.provider, query, result.hits)
            identifier, message = save_turn(state.sessions, data.conversation_id, previous, data.question, answer)
        except ConversationMissing:
            raise HTTPException(404, "Conversation not found.") from None
        except ConversationChanged:
            raise HTTPException(409, "The conversation changed while generating this reply. Reload and retry.") from None
    return ChatView(conversation_id=identifier, search_query=query, message=message, retrieval=result)


@api.post("/conversations", response_model=ConversationView, status_code=201, tags=["Conversations"])
def new_conversation(request: Request, data: ConversationInput):
    with request.app.state.sessions.begin() as session:
        row = Conversation(title=data.title)
        session.add(row)
        session.flush()
        return ConversationView.model_validate(row)


@api.get("/conversations", response_model=list[ConversationView], tags=["Conversations"])
def conversations(request: Request, limit: PageSize = 50, offset: Offset = 0):
    with request.app.state.sessions() as session:
        return [ConversationView.model_validate(row) for row in session.scalars(
            select(Conversation).order_by(Conversation.updated_at.desc(), Conversation.id).offset(offset).limit(limit)
        )]


@api.get("/conversations/{identifier}", response_model=ConversationHistory, tags=["Conversations"])
def conversation_history(request: Request, identifier: UUID, limit: PageSize = 50,
                         before: Annotated[int | None, Query(ge=1)] = None):
    with request.app.state.sessions() as session:
        row = session.get(Conversation, identifier)
        if row is None:
            raise HTTPException(404, "Conversation not found.")
        statement = select(Message).where(Message.conversation_id == identifier).order_by(Message.id.desc()).limit(limit + 1)
        if before is not None:
            statement = statement.where(Message.id < before)
        messages = list(session.scalars(statement))
        return ConversationHistory(**ConversationView.model_validate(row).model_dump(),
                                   messages=list(reversed(messages[:limit])), has_older=len(messages) > limit)


@api.delete("/conversations/{identifier}", status_code=204, tags=["Conversations"])
def remove_conversation(request: Request, identifier: UUID):
    with request.app.state.sessions.begin() as session:
        row = session.get(Conversation, identifier, with_for_update=True)
        if row is None:
            raise HTTPException(404, "Conversation not found.")
        session.delete(row)
    return Response(status_code=204)


@api.get("/evaluations/demo", tags=["Evaluation"], response_class=Response)
def download_demo():
    return Response(demo_bytes(), media_type="text/plain", headers={
        "Content-Disposition": 'attachment; filename="handbook.txt"',
    })


@api.post("/evaluations", response_model=EvaluationView, status_code=201, tags=["Evaluation"])
def create_evaluation(request: Request, data: EvaluationInput):
    state = request.app.state
    settings = state.settings
    state.vectors.require_client()
    if not settings.rerank_enabled:
        raise HTTPException(422, "Enable reranking to compare all four retrieval modes.")
    if data.top_k > settings.rerank_top_k:
        raise HTTPException(422, f"Evaluation top_k must be at most {settings.rerank_top_k}.")
    if data.judge_answers and not settings.deepseek_configured:
        raise ConfigurationError("Set DEEPSEEK_API_KEY in .env to score answers.")
    with limited(state.evaluation_gate):
        with state.sessions() as session:
            if data.cases is None:
                demo_id = session.scalar(select(Document.id).where(
                    Document.sha256 == demo_digest(), Document.status == "ready"
                ))
                if demo_id is None:
                    raise HTTPException(409, "Upload the demo handbook and wait for it to become ready.")
                scope = [demo_id]
            else:
                scope = ready_scope(session, data.document_ids)
                if set(scope) != set(data.document_ids or []):
                    raise HTTPException(409, "All evaluation documents must be ready.")
        result = evaluate(state.vectors, state.provider, settings, data.cases or demo_cases(), scope,
                          data.top_k, data.judge_answers, "custom" if data.cases else "demo")
        with state.sessions.begin() as session:
            row = EvaluationRun(report=result)
            session.add(row)
            session.flush()
            return EvaluationView.model_validate(row)


@api.get("/evaluations", response_model=list[EvaluationView], tags=["Evaluation"])
def evaluations(request: Request, limit: PageSize = 10, offset: Offset = 0):
    with request.app.state.sessions() as session:
        return [EvaluationView.model_validate(row) for row in session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc(), EvaluationRun.id).limit(limit).offset(offset)
        )]
