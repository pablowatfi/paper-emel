from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from arxiv_rag.models import Chunk, IngestState, Paper, RAGResponse, RetrievalResult


def _paper(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "arxiv_id": "2401.12345",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture...",
        "authors": ["Alice", "Bob"],
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "cs.AI"],
        "published_at": datetime(2024, 1, 15, tzinfo=UTC),
        "updated_at": datetime(2024, 1, 16, tzinfo=UTC),
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "abs_url": "https://arxiv.org/abs/2401.12345",
    }
    base.update(overrides)
    return base


def _chunk(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "chunk_id": "2401.12345::title_abstract::0",
        "arxiv_id": "2401.12345",
        "text": "Attention Is All You Need. We propose...",
        "section": "title_abstract",
        "token_count": 42,
        "paper_metadata": {
            "title": "Attention Is All You Need",
            "authors": ["Alice", "Bob"],
            "published_at": "2024-01-15T00:00:00+00:00",
            "primary_category": "cs.LG",
            "abs_url": "https://arxiv.org/abs/2401.12345",
        },
    }
    base.update(overrides)
    return base


class TestPaper:
    def test_valid_construction(self) -> None:
        paper = Paper(**_paper())  # type: ignore[arg-type]
        assert paper.arxiv_id == "2401.12345"
        assert paper.authors == ["Alice", "Bob"]

    def test_arxiv_id_with_version(self) -> None:
        paper = Paper(**_paper(arxiv_id="2401.12345v3"))  # type: ignore[arg-type]
        assert paper.arxiv_id == "2401.12345v3"

    def test_arxiv_id_five_digit(self) -> None:
        paper = Paper(**_paper(arxiv_id="2401.99999"))  # type: ignore[arg-type]
        assert paper.arxiv_id == "2401.99999"

    def test_invalid_arxiv_id_format(self) -> None:
        with pytest.raises(ValidationError):
            Paper(**_paper(arxiv_id="invalid-id"))  # type: ignore[arg-type]

    def test_invalid_arxiv_id_short(self) -> None:
        with pytest.raises(ValidationError):
            Paper(**_paper(arxiv_id="2401.123"))  # type: ignore[arg-type]

    def test_missing_required_field(self) -> None:
        data = _paper()
        del data["title"]
        with pytest.raises(ValidationError):
            Paper(**data)  # type: ignore[arg-type]

    def test_empty_authors_allowed(self) -> None:
        paper = Paper(**_paper(authors=[]))  # type: ignore[arg-type]
        assert paper.authors == []


class TestChunk:
    def test_valid_construction(self) -> None:
        chunk = Chunk(**_chunk())  # type: ignore[arg-type]
        assert chunk.section == "title_abstract"
        assert chunk.token_count == 42

    def test_metadata_section(self) -> None:
        chunk = Chunk(**_chunk(section="metadata", chunk_id="2401.12345::metadata::0"))  # type: ignore[arg-type]
        assert chunk.section == "metadata"

    def test_invalid_section(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(**_chunk(section="full_text"))  # type: ignore[arg-type]

    def test_missing_chunk_id(self) -> None:
        data = _chunk()
        del data["chunk_id"]
        with pytest.raises(ValidationError):
            Chunk(**data)  # type: ignore[arg-type]


class TestRetrievalResult:
    def test_valid_construction(self) -> None:
        chunk = Chunk(**_chunk())  # type: ignore[arg-type]
        result = RetrievalResult(
            chunk=chunk,
            dense_score=0.95,
            sparse_score=0.80,
            rrf_score=0.033,
            rank=1,
        )
        assert result.rank == 1
        assert result.chunk.arxiv_id == "2401.12345"


class TestRAGResponse:
    def _make_source(self) -> RetrievalResult:
        return RetrievalResult(
            chunk=Chunk(**_chunk()),  # type: ignore[arg-type]
            dense_score=0.9,
            sparse_score=0.8,
            rrf_score=0.03,
            rank=1,
        )

    def test_full_mode(self) -> None:
        resp = RAGResponse(
            query="transformers?",
            answer="Transformers use attention.",
            sources=[self._make_source()],
            mode="full",
            latency_ms={"retrieval": 45.0, "llm": 1200.0, "total": 1245.0},
            trace_id="trace-abc-123",
        )
        assert resp.mode == "full"
        assert resp.cached is False

    def test_degraded_mode_answer_none(self) -> None:
        resp = RAGResponse(
            query="transformers?",
            answer=None,
            sources=[self._make_source()],
            mode="degraded",
            latency_ms={"retrieval": 45.0, "total": 45.0},
            trace_id="trace-abc-456",
        )
        assert resp.answer is None
        assert resp.mode == "degraded"

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            RAGResponse(
                query="q",
                answer="a",
                sources=[],
                mode="unknown",  # type: ignore[arg-type]
                latency_ms={},
                trace_id="t",
            )


class TestIngestState:
    def test_valid_construction(self) -> None:
        state = IngestState(
            last_run_utc=datetime(2024, 1, 15, tzinfo=UTC),
            last_oai_token=None,
            papers_in_index=9500,
            last_paper_published_at=datetime(2024, 1, 14, tzinfo=UTC),
        )
        assert state.papers_in_index == 9500
        assert state.last_oai_token is None

    def test_with_resumption_token(self) -> None:
        state = IngestState(
            last_run_utc=datetime(2024, 1, 15, tzinfo=UTC),
            last_oai_token="token-abc-xyz",
            papers_in_index=100,
            last_paper_published_at=datetime(2024, 1, 14, tzinfo=UTC),
        )
        assert state.last_oai_token == "token-abc-xyz"
