"""Hybrid (dense + sparse) Qdrant vector store."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    PayloadSchemaType,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from arxiv_rag.models import Chunk

_UPSERT_BATCH = 100
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"
_DIMS = 384


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _pub_ts(paper_metadata: dict[str, object]) -> float:
    raw = str(paper_metadata.get("published_at", "2000-01-01T00:00:00+00:00"))
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _build_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "arxiv_id": chunk.arxiv_id,
        "text": chunk.text,
        "section": chunk.section,
        "token_count": chunk.token_count,
        "paper_metadata": dict(chunk.paper_metadata),
        "primary_category": str(chunk.paper_metadata.get("primary_category", "")),
        "published_at_ts": _pub_ts(chunk.paper_metadata),
    }


class QdrantStore:
    """Async wrapper around Qdrant for hybrid dense+sparse ingestion."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection: str = "arxiv_cs_lg",
    ) -> None:
        self._collection = collection
        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)

    async def __aenter__(self) -> QdrantStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def setup_collection(self) -> None:
        """Create collection + payload indexes (idempotent)."""
        if await self._client.collection_exists(self._collection):
            return

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={_DENSE_NAME: VectorParams(size=_DIMS, distance=Distance.COSINE)},
            sparse_vectors_config={_SPARSE_NAME: SparseVectorParams()},
        )

        for field, schema in [
            ("arxiv_id", PayloadSchemaType.KEYWORD),
            ("primary_category", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("published_at_ts", PayloadSchemaType.FLOAT),
        ]:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=schema,
            )

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vecs: npt.NDArray[np.float32],
        sparse_vecs: list[SparseVector],
    ) -> None:
        """Bulk upsert chunks with dense and sparse vectors."""
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    _DENSE_NAME: dense_vecs[i].tolist(),
                    _SPARSE_NAME: sparse_vecs[i],
                },
                payload=_build_payload(chunk),
            )
            for i, chunk in enumerate(chunks)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _UPSERT_BATCH],
            )

    async def evict_before(self, cutoff: datetime) -> int:
        """Delete points published before cutoff; return count deleted."""
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        eviction_filter = Filter(
            must=[
                FieldCondition(
                    key="published_at_ts",
                    range=Range(lt=cutoff.timestamp()),
                )
            ]
        )
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=eviction_filter,
            exact=True,
        )
        n = int(result.count)
        if n > 0:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=eviction_filter),
            )
        return n

    async def count(self) -> int:
        result = await self._client.count(self._collection, exact=True)
        return int(result.count)

    async def health_check(self) -> bool:
        try:
            await self._client.get_collection(self._collection)
            return True
        except Exception:
            return False
