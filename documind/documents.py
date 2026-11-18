import hashlib
import logging
import threading
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import ConfigurationError, Settings
from .database import Document
from .ingestion import DocumentError, extract, split_document
from .retrieval import RetrievalError, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Upload:
    filename: str
    extension: str
    content: bytes


def enqueue(session: Session, upload: Upload) -> tuple[Document, bool]:
    digest = hashlib.sha256(upload.content).hexdigest()
    statement = insert(Document).values(
        filename=upload.filename, extension=upload.extension, content=upload.content,
        size_bytes=len(upload.content), sha256=digest,
    ).on_conflict_do_nothing(index_elements=[Document.sha256]).returning(Document)
    row = session.scalars(statement).first()
    if row is not None:
        return row, False
    duplicate = session.scalar(select(Document).where(Document.sha256 == digest))
    if duplicate is None:
        raise DocumentError("An identical upload changed concurrently. Retry the upload.")
    return duplicate, True


class DocumentWorker:
    """Drive persistent jobs with a PostgreSQL advisory lock per worker loop."""

    def __init__(self, engine: Engine, sessions: sessionmaker, store: VectorStore, settings: Settings):
        self.engine = engine
        self.sessions = sessions
        self.store = store
        self.settings = settings
        self.stop = threading.Event()
        self.wakeup = threading.Event()
        self.thread = threading.Thread(target=self.run, name="document-worker", daemon=True)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                found_job = self.process_next()
            except Exception as exc:
                logger.error("document_worker_error type=%s", type(exc).__name__)
                found_job = False
            if not found_job:
                self.wakeup.wait(2)
                self.wakeup.clear()

    def close(self) -> None:
        self.stop.set()
        self.wakeup.set()
        self.thread.join(timeout=5)

    def process_next(self) -> bool:
        if not self.settings.qdrant_configured:
            return False
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as lock_connection:
            acquired = lock_connection.scalar(text("SELECT pg_try_advisory_lock(1398892041)"))
            if not acquired:
                return False
            try:
                return self.process_locked_job()
            finally:
                try:
                    lock_connection.execute(text("SELECT pg_advisory_unlock(1398892041)"))
                except Exception:
                    lock_connection.invalidate()
                    raise

    def process_locked_job(self) -> bool:
        with self.sessions.begin() as session:
            job = session.scalar(
                select(Document).where(Document.status.in_(("queued", "processing", "deleting")))
                .order_by(Document.created_at, Document.id).limit(1).with_for_update()
            )
            if job is None:
                return False
            identifier = job.id
            deleting = job.status == "deleting"
            if not deleting:
                job.status = "processing"
                upload = Upload(job.filename, job.extension, job.content or b"")
        try:
            if deleting:
                self.delete_job(identifier)
            else:
                self.index_job(identifier, upload)
        except (ConfigurationError, DocumentError, RetrievalError) as exc:
            self.fail(identifier, deleting, str(exc))
        except Exception as exc:
            logger.error("document_job_error id=%s type=%s", identifier, type(exc).__name__)
            self.fail(identifier, deleting, "An unexpected processing error occurred. Retry this document.")
        return True

    def delete_job(self, identifier: UUID) -> None:
        self.store.delete(identifier)
        with self.sessions.begin() as session:
            if (row := session.get(Document, identifier)) is not None:
                session.delete(row)
        logger.info("document_deleted id=%s", identifier)

    def index_job(self, identifier: UUID, upload: Upload) -> None:
        extracted = extract(upload.content, upload.extension)
        chunks = split_document(identifier, extracted, self.settings.chunk_size, self.settings.chunk_overlap)
        self.store.index(identifier, upload.filename, chunks)
        with self.sessions.begin() as session:
            row = session.get(Document, identifier, with_for_update=True)
            if row is not None and row.status == "processing":
                row.status = "ready"
                row.error = None
                row.chunk_count = len(chunks)
                row.page_count = extracted.page_count
                row.details = extracted.details
                row.content = None
        logger.info("document_indexed id=%s chunks=%s", identifier, len(chunks))

    def fail(self, identifier: UUID, deleting: bool, reason: str) -> None:
        expected = "deleting" if deleting else "processing"
        with self.sessions.begin() as session:
            row = session.get(Document, identifier, with_for_update=True)
            if row is not None and row.status == expected:
                row.status = "deletion_failed" if deleting else "failed"
                row.error = reason
        logger.warning("document_job_failed id=%s operation=%s", identifier, expected)


def ready_scope(session: Session, selected: list[UUID] | None) -> list[UUID]:
    statement = select(Document.id).where(Document.status == "ready")
    if selected is not None:
        statement = statement.where(Document.id.in_(selected))
    return list(session.scalars(statement))


def document_summary(session: Session) -> dict:
    rows = session.execute(
        select(Document.status, func.count(), func.coalesce(func.sum(Document.chunk_count), 0)).group_by(Document.status)
    )
    by_status: dict[str, int] = {}
    ready_chunks = 0
    for status, count, chunks in rows:
        by_status[status] = count
        if status == "ready":
            ready_chunks = chunks
    return {"total_documents": sum(by_status.values()), "ready_documents": by_status.get("ready", 0),
            "failed_documents": by_status.get("failed", 0) + by_status.get("deletion_failed", 0),
            "indexed_chunks": ready_chunks}
