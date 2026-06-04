from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="arxiv_cs_lg")

    # LLM providers
    groq_api_key: str = Field(default="")
    cerebras_api_key: str = Field(default="")

    # Redis / cache (optional — graceful degradation if missing)
    upstash_redis_url: str = Field(default="")
    upstash_redis_token: str = Field(default="")

    # Observability (optional)
    langfuse_secret_key: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    sentry_dsn: str = Field(default="")

    # Tunables
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    rate_limit_per_day: int = Field(default=10)
    corpus_window_days: int = Field(default=90)


def get_settings() -> Settings:
    return Settings()
