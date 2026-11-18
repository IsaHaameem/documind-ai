from functools import lru_cache
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


class ConfigurationError(Exception):
    """Configuration that must be supplied before an operation can run."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True, hide_input_in_errors=True
    )

    app_name: str = "DocuMind AI"
    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: SecretStr = SecretStr("")
    cors_origins: list[str] = Field(default_factory=list)
    postgres_db: str = "documind"
    postgres_user: str = "documind_user"
    postgres_password: SecretStr = SecretStr("")
    database_url: SecretStr | None = None
    redis_url: SecretStr = SecretStr("redis://redis:6379/0")
    embedding_cache_ttl_seconds: int = Field(3600, ge=0, le=86400)

    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection_name: str = Field("documind_documents", pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    embedding_model: str = EMBEDDING_MODEL
    reranker_model: str = RERANKER_MODEL
    embedding_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    reranker_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    rerank_enabled: bool = True
    embedding_batch_size: int = Field(32, ge=1, le=128)
    retrieval_top_k: int = Field(10, ge=1, le=50)
    rerank_top_k: int = Field(5, ge=1, le=20)
    dense_min_score: float = Field(0.35, ge=-1, le=1)

    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_temperature: float = Field(0.2, ge=0, le=2)
    deepseek_timeout_seconds: float = Field(60, gt=0, le=180)
    max_upload_size_mb: int = Field(50, ge=1, le=100)
    max_upload_files: int = Field(8, ge=1, le=20)
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(150, ge=0)

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError("RERANK_TOP_K must be at most RETRIEVAL_TOP_K.")
        if self.embedding_model != EMBEDDING_MODEL and self.embedding_revision is None:
            raise ValueError("Set EMBEDDING_REVISION to the Hugging Face commit for custom embedding models.")
        if self.rerank_enabled and self.reranker_model != RERANKER_MODEL and self.reranker_revision is None:
            raise ValueError("Set RERANKER_REVISION to the Hugging Face commit for custom rerankers.")
        if self.qdrant_url:
            self.check_https(self.qdrant_url, "QDRANT_URL")
        self.check_https(self.deepseek_base_url, "DEEPSEEK_BASE_URL")
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must contain explicit origins.")
        return self

    @staticmethod
    def check_https(url: str, name: str) -> None:
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(f"{name} must be an HTTPS URL without credentials, query or fragment.")

    def db_url(self) -> URL:
        if self.database_url:
            try:
                url = make_url(self.database_url.get_secret_value())
            except Exception:
                raise ConfigurationError("DATABASE_URL is invalid.") from None
        else:
            password = self.postgres_password.get_secret_value()
            if not password or password.startswith("CHANGE_ME"):
                raise ConfigurationError("Set POSTGRES_PASSWORD in .env.")
            url = URL.create("postgresql+psycopg", self.postgres_user, password, "postgres", 5432, self.postgres_db)
        if (url.drivername != "postgresql+psycopg" or url.host != "postgres"
                or url.port not in (None, 5432) or url.query):
            raise ConfigurationError("DATABASE_URL must use postgresql+psycopg and Docker postgres:5432.")
        if not url.password or url.password.startswith("CHANGE_ME"):
            raise ConfigurationError("Set a real PostgreSQL password in .env.")
        return url

    @property
    def workspace_enabled(self) -> bool:
        value = self.secret_key.get_secret_value()
        return (len(value) >= 32 and value.isascii() and not value.startswith("CHANGE_ME")
                and all(character.isprintable() and not character.isspace() for character in value))

    @property
    def embedding_commit(self) -> str:
        return self.embedding_revision or EMBEDDING_REVISION

    @property
    def reranker_commit(self) -> str:
        return self.reranker_revision or RERANKER_REVISION

    @property
    def qdrant_configured(self) -> bool:
        key = self.qdrant_api_key.get_secret_value()
        return bool(self.qdrant_url and key and not key.startswith("CHANGE_ME"))

    @property
    def deepseek_configured(self) -> bool:
        key = self.deepseek_api_key.get_secret_value()
        return bool(key and not key.startswith("CHANGE_ME"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
