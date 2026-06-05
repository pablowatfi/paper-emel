"""Unit tests for the Paper → Chunk converter."""

from __future__ import annotations

from datetime import UTC, datetime

import tiktoken

from arxiv_rag.ingestion.chunker import chunk_paper
from arxiv_rag.models import Paper

_ENC = tiktoken.get_encoding("cl100k_base")


def _make_paper(**overrides: object) -> Paper:
    base: dict[str, object] = {
        "arxiv_id": "2401.12345",
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer, a novel model architecture.",
        "authors": ["Alice Smith", "Bob Jones", "Carol White"],
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "cs.AI"],
        "published_at": datetime(2024, 1, 15, tzinfo=UTC),
        "updated_at": datetime(2024, 1, 16, tzinfo=UTC),
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "abs_url": "https://arxiv.org/abs/2401.12345",
    }
    base.update(overrides)
    return Paper(**base)  # type: ignore[arg-type]


class TestChunkPaper:
    def test_returns_exactly_two_chunks(self) -> None:
        assert len(chunk_paper(_make_paper())) == 2

    def test_both_sections_present(self) -> None:
        sections = {c.section for c in chunk_paper(_make_paper())}
        assert sections == {"title_abstract", "metadata"}

    def test_chunk_ids_deterministic(self) -> None:
        paper = _make_paper()
        assert [c.chunk_id for c in chunk_paper(paper)] == [c.chunk_id for c in chunk_paper(paper)]

    def test_chunk_id_format(self) -> None:
        chunks = chunk_paper(_make_paper())
        ta = next(c for c in chunks if c.section == "title_abstract")
        md = next(c for c in chunks if c.section == "metadata")
        assert ta.chunk_id == "2401.12345::title_abstract::0"
        assert md.chunk_id == "2401.12345::metadata::0"

    def test_title_abstract_contains_title_and_abstract(self) -> None:
        paper = _make_paper()
        ta = next(c for c in chunk_paper(paper) if c.section == "title_abstract")
        assert paper.title in ta.text
        assert paper.abstract in ta.text

    def test_token_count_matches_tiktoken(self) -> None:
        paper = _make_paper()
        for chunk in chunk_paper(paper):
            expected = len(_ENC.encode(chunk.text))
            assert chunk.token_count == expected

    def test_paper_metadata_fields_present(self) -> None:
        paper = _make_paper()
        for chunk in chunk_paper(paper):
            m = chunk.paper_metadata
            assert m["title"] == paper.title
            assert m["authors"] == paper.authors
            assert m["primary_category"] == paper.primary_category
            assert m["abs_url"] == paper.abs_url
            assert isinstance(m["published_at"], str)

    def test_no_authors_metadata_chunk_still_valid(self) -> None:
        paper = _make_paper(authors=[])
        md = next(c for c in chunk_paper(paper) if c.section == "metadata")
        assert "unknown authors" in md.text
        assert md.token_count > 0

    def test_arxiv_id_propagated_to_all_chunks(self) -> None:
        for chunk in chunk_paper(_make_paper()):
            assert chunk.arxiv_id == "2401.12345"

    def test_many_authors_uses_et_al(self) -> None:
        paper = _make_paper(authors=["Alice", "Bob", "Carol", "Dave"])
        md = next(c for c in chunk_paper(paper) if c.section == "metadata")
        assert "et al." in md.text
        assert "Alice" in md.text

    def test_metadata_contains_date_and_category(self) -> None:
        paper = _make_paper()
        md = next(c for c in chunk_paper(paper) if c.section == "metadata")
        assert "2024-01-15" in md.text
        assert "cs.LG" in md.text
