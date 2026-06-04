from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A paper as fetched from arXiv OAI-PMH."""

    arxiv_id: str = Field(..., pattern=r"^\d{4}\.\d{4,5}(v\d+)?$")
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published_at: datetime
    updated_at: datetime
    pdf_url: str
    abs_url: str


class Chunk(BaseModel):
    """A unit of text indexed in the vector store."""

    chunk_id: str  # f"{arxiv_id}::{section}::{idx}"
    arxiv_id: str
    text: str
    section: Literal["title_abstract", "metadata"]
    token_count: int
    paper_metadata: dict[str, object]  # title, authors, published_at, primary_category, abs_url


class RetrievalResult(BaseModel):
    """A single retrieved chunk with scoring breakdown."""

    chunk: Chunk
    dense_score: float
    sparse_score: float
    rrf_score: float
    rank: int


class RAGResponse(BaseModel):
    """Final response returned to the user."""

    query: str
    answer: str | None  # None in degraded mode
    sources: list[RetrievalResult]
    mode: Literal["full", "degraded"]
    latency_ms: dict[str, float]  # {"retrieval": 45, "llm": 1200, "total": 1245}
    trace_id: str
    cached: bool = False


class IngestState(BaseModel):
    """Tracks ingestion checkpoint, committed to repo as ingest_state.json."""

    last_run_utc: datetime
    last_oai_token: str | None
    papers_in_index: int
    last_paper_published_at: datetime
