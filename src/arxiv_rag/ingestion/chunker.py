"""Converts a Paper into indexable Chunks."""

from __future__ import annotations

import tiktoken

from arxiv_rag.models import Chunk, Paper

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _authors_short(authors: list[str]) -> str:
    if not authors:
        return "unknown authors"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def _paper_metadata(paper: Paper) -> dict[str, object]:
    meta: dict[str, object] = {
        "title": paper.title,
        "authors": paper.authors,
        "published_at": paper.published_at.isoformat(),
        "primary_category": paper.primary_category,
        "abs_url": paper.abs_url,
    }
    return meta


def chunk_paper(paper: Paper) -> list[Chunk]:
    """Return exactly 2 Chunks for a paper: title_abstract and metadata."""
    meta = _paper_metadata(paper)

    title_abstract_text = f"{paper.title}\n\n{paper.abstract}"

    date_str = paper.published_at.strftime("%Y-%m-%d")
    cats = ", ".join(paper.categories) if paper.categories else paper.primary_category
    metadata_text = (
        f"Paper '{paper.title}' by {_authors_short(paper.authors)} "
        f"published on {date_str} in categories {cats}. "
        f"Primary topic: {paper.primary_category}."
    )

    return [
        Chunk(
            chunk_id=f"{paper.arxiv_id}::title_abstract::0",
            arxiv_id=paper.arxiv_id,
            text=title_abstract_text,
            section="title_abstract",
            token_count=_token_count(title_abstract_text),
            paper_metadata=meta,
        ),
        Chunk(
            chunk_id=f"{paper.arxiv_id}::metadata::0",
            arxiv_id=paper.arxiv_id,
            text=metadata_text,
            section="metadata",
            token_count=_token_count(metadata_text),
            paper_metadata=meta,
        ),
    ]
