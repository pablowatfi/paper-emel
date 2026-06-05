"""Integration tests for QdrantStore against local Docker Qdrant.

Requires: docker compose -f infra/docker-compose.yml up -d
Run with: make test-int
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from arxiv_rag.indexing.qdrant_store import QdrantStore
from arxiv_rag.indexing.sparse_encoder import SparseEncoder
from arxiv_rag.models import Chunk

_URL = "http://localhost:6333"
_COLLECTION = "test_qdrant_store_integration"
_ENC = SparseEncoder()


def _chunks(n: int, id_prefix: str = "2401", days_old: int = 10) -> list[Chunk]:
    pub_at = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    return [
        Chunk(
            chunk_id=f"{id_prefix}.{i:05d}::title_abstract::0",
            arxiv_id=f"{id_prefix}.{i:05d}",
            text=f"Paper {id_prefix} number {i} about machine learning transformers.",
            section="title_abstract",
            token_count=12,
            paper_metadata={
                "title": f"Paper {i}",
                "authors": ["Alice"],
                "published_at": pub_at,
                "primary_category": "cs.LG",
                "abs_url": f"https://arxiv.org/abs/{id_prefix}.{i:05d}",
            },
        )
        for i in range(n)
    ]


def _random_dense(n: int) -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, 384)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture(scope="module")
async def store() -> AsyncIterator[QdrantStore]:
    from qdrant_client import AsyncQdrantClient

    # Ensure clean slate
    client = AsyncQdrantClient(url=_URL)
    if await client.collection_exists(_COLLECTION):
        await client.delete_collection(_COLLECTION)
    await client.close()

    s = QdrantStore(url=_URL, collection=_COLLECTION)
    await s.setup_collection()
    yield s

    # Teardown
    await s._client.delete_collection(_COLLECTION)
    await s.close()


async def test_health_check(store: QdrantStore) -> None:
    assert await store.health_check() is True


async def test_setup_idempotent(store: QdrantStore) -> None:
    """Calling setup_collection twice must not raise."""
    await store.setup_collection()


async def test_upsert_and_count(store: QdrantStore) -> None:
    batch = _chunks(10, id_prefix="2401", days_old=10)
    await store.upsert_chunks(batch, _random_dense(10), _ENC.encode_batch([c.text for c in batch]))
    assert await store.count() == 10


async def test_upsert_idempotent(store: QdrantStore) -> None:
    """Re-upserting same chunk_ids must not grow the collection."""
    batch = _chunks(10, id_prefix="2401", days_old=10)
    await store.upsert_chunks(batch, _random_dense(10), _ENC.encode_batch([c.text for c in batch]))
    assert await store.count() == 10


async def test_eviction(store: QdrantStore) -> None:
    # Add 5 stale chunks (100 days old) alongside the existing 10 (10 days old)
    old = _chunks(5, id_prefix="2402", days_old=100)
    await store.upsert_chunks(old, _random_dense(5), _ENC.encode_batch([c.text for c in old]))
    assert await store.count() == 15

    cutoff = datetime.now(UTC) - timedelta(days=30)
    evicted = await store.evict_before(cutoff)

    assert evicted == 5
    assert await store.count() == 10
