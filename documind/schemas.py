from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

from .evaluation import EvaluationCase
from .generation import Citation
from .retrieval import SearchMode

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1500)]


class DocumentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    extension: str
    size_bytes: int
    status: str
    chunk_count: int
    page_count: int | None
    details: dict[str, str]
    error: str | None
    created_at: datetime
    updated_at: datetime


class UploadResult(BaseModel):
    filename: str
    document: DocumentView | None = None
    duplicate: bool = False
    error: str | None = None


class Dashboard(BaseModel):
    total_documents: int
    ready_documents: int
    failed_documents: int
    indexed_chunks: int
    recent: list[DocumentView]


class SearchInput(BaseModel):
    question: Question
    document_ids: list[UUID] | None = Field(default=None, max_length=100)
    mode: SearchMode = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=50)
    dense_min_score: float | None = Field(default=None, ge=-1, le=1)
    rerank: bool | None = None


class HitView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID
    document_id: UUID
    filename: str
    location: str
    page: int | None
    text: str
    retrieval_score: float
    rerank_score: float | None


class RetrievalView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidates: list[HitView]
    hits: list[HitView]
    latency_ms: dict[str, float]
    embedding_cache_hit: bool


class ConversationInput(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)] = "New conversation"


class ConversationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation]
    insufficient_context: bool
    created_at: datetime


class ConversationHistory(ConversationView):
    messages: list[MessageView]
    has_older: bool


class ChatInput(SearchInput):
    conversation_id: UUID | None = None


class ChatView(BaseModel):
    conversation_id: UUID
    search_query: str
    message: MessageView
    retrieval: RetrievalView


class EvaluationInput(BaseModel):
    cases: list[EvaluationCase] | None = Field(default=None, min_length=1, max_length=12)
    document_ids: list[UUID] | None = Field(default=None, min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)
    judge_answers: bool = False

    @model_validator(mode="after")
    def validate_custom_scope(self):
        if self.cases is not None and not self.document_ids:
            raise ValueError("Select ready documents for a custom evaluation.")
        if self.judge_answers and self.cases and len(self.cases) > 4:
            raise ValueError("Answer judging is limited to four cases per evaluation.")
        return self


class EvaluationCaseResult(BaseModel):
    question: str
    expected_answer: str
    recall_at_k: float
    mrr: float
    context_precision: float
    context_recall: float
    latency_ms: float
    retrieved_chunk_ids: list[UUID]
    answer_faithfulness: float | None = None
    answer_relevance: float | None = None
    answer: str | None = None
    judge_status: str | None = None
    judge_error: str | None = None


class EvaluationModeResult(BaseModel):
    averages: dict[str, float]
    cases: list[EvaluationCaseResult]


class EvaluationReport(BaseModel):
    dataset: Literal["demo", "custom"]
    k: int
    judge_answers: bool
    modes: dict[str, EvaluationModeResult]


class EvaluationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    report: EvaluationReport
