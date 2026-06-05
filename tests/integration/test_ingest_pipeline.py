"""Integration tests for the ingestion pipeline in fixture mode.

Requires: docker compose -f infra/docker-compose.yml up -d
Run with: make test-int
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from arxiv_rag.ingestion.pipeline import run_ingest

_URL = "http://localhost:6333"
_COLLECTION = "test_ingest_pipeline"


@pytest.fixture(scope="module")
async def clean_collection() -> AsyncIterator[None]:
    client = AsyncQdrantClient(url=_URL)
    if await client.collection_exists(_COLLECTION):
        await client.delete_collection(_COLLECTION)
    await client.close()
    yield
    client = AsyncQdrantClient(url=_URL)
    if await client.collection_exists(_COLLECTION):
        await client.delete_collection(_COLLECTION)
    await client.close()


async def test_fixture_ingest_populates_qdrant(tmp_path: Path, clean_collection: None) -> None:
    state_path = tmp_path / "ingest_state.json"

    state = await run_ingest(
        fixture=True,
        state_path=state_path,
        qdrant_url=_URL,
        collection=_COLLECTION,
    )

    assert state_path.exists(), "state file must be written"
    assert state.papers_in_index > 0, "at least one chunk must be indexed"
    # 200 papers × 2 chunks = 400 expected
    assert state.papers_in_index == 400
    assert state.last_run_utc is not None
    assert state.last_paper_published_at.year >= 2024


async def test_fixture_ingest_is_idempotent(tmp_path: Path, clean_collection: None) -> None:
    state_path = tmp_path / "ingest_state.json"

    state1 = await run_ingest(
        fixture=True,
        state_path=state_path,
        qdrant_url=_URL,
        collection=_COLLECTION,
    )
    state2 = await run_ingest(
        fixture=True,
        state_path=state_path,
        qdrant_url=_URL,
        collection=_COLLECTION,
    )

    assert state1.papers_in_index == state2.papers_in_index
