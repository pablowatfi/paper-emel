import pytest

from arxiv_rag.settings import Settings, get_settings


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.qdrant_url == "http://localhost:6333"
        assert s.qdrant_collection == "arxiv_cs_lg"
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"
        assert s.rate_limit_per_day == 10
        assert s.corpus_window_days == 90
        assert s.langfuse_host == "https://cloud.langfuse.com"

    def test_optional_fields_default_empty(self) -> None:
        s = Settings()
        assert s.groq_api_key == ""
        assert s.cerebras_api_key == ""
        assert s.upstash_redis_url == ""
        assert s.langfuse_secret_key == ""
        assert s.sentry_dsn == ""

    def test_overrides_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_COLLECTION", "my_collection")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
        monkeypatch.setenv("RATE_LIMIT_PER_DAY", "25")
        s = Settings()
        assert s.qdrant_url == "http://qdrant.example.com:6333"
        assert s.qdrant_collection == "my_collection"
        assert s.groq_api_key == "gsk_test_key"
        assert s.rate_limit_per_day == 25

    def test_get_settings_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_extra_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOTALLY_UNKNOWN_VAR", "ignored")
        s = Settings()
        assert not hasattr(s, "totally_unknown_var")
