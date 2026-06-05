"""End-to-end ingestion pipeline: OAI-PMH → chunk → embed → upsert → evict."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from arxiv_rag.indexing.embedder import BGEEmbedder
from arxiv_rag.indexing.qdrant_store import QdrantStore
from arxiv_rag.indexing.sparse_encoder import SparseEncoder
from arxiv_rag.ingestion.chunker import chunk_paper
from arxiv_rag.ingestion.oai_client import OAIClient
from arxiv_rag.models import Chunk, IngestState, Paper
from arxiv_rag.settings import get_settings

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_STATE_PATH = _PROJECT_ROOT / "ingest_state.json"
_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "sample_papers_200.jsonl"
_PIPELINE_BATCH = 128  # chunks to embed + upsert per round-trip


def _load_state(path: Path) -> IngestState:
    if path.exists():
        return IngestState.model_validate_json(path.read_text())
    epoch = datetime(2000, 1, 1, tzinfo=UTC)
    return IngestState(
        last_run_utc=epoch,
        last_oai_token=None,
        papers_in_index=0,
        last_paper_published_at=epoch,
    )


def _save_state(state: IngestState, path: Path) -> None:
    path.write_text(state.model_dump_json(indent=2))


def _load_fixture() -> list[Paper]:
    papers: list[Paper] = []
    with _FIXTURE_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                papers.append(Paper.model_validate_json(line))
    return papers


async def _flush(
    chunks: list[Chunk],
    embedder: BGEEmbedder,
    sparse_enc: SparseEncoder,
    store: QdrantStore,
) -> None:
    texts = [c.text for c in chunks]
    dense = embedder.embed_texts(texts)
    sparse = sparse_enc.encode_batch(texts)
    await store.upsert_chunks(chunks, dense, sparse)


async def run_ingest(
    days_back: int = 90,
    evict_older_than_days: int = 90,
    fixture: bool = False,
    state_path: Path = _STATE_PATH,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection: str | None = None,
) -> IngestState:
    """Run one full ingest cycle; return the updated IngestState."""
    settings = get_settings()
    url = qdrant_url or settings.qdrant_url
    key = qdrant_api_key or settings.qdrant_api_key
    coll = collection or settings.qdrant_collection

    store = QdrantStore(url=url, api_key=key, collection=coll)
    embedder = BGEEmbedder()
    sparse_enc = SparseEncoder()

    state = _load_state(state_path)
    now = datetime.now(UTC)
    window_start = now - timedelta(days=days_back)
    from_date = max(state.last_run_utc, window_start)
    cutoff = now - timedelta(days=evict_older_than_days)

    await store.setup_collection()

    # --- Harvest ---
    papers: list[Paper]
    if fixture:
        papers = _load_fixture()
        log.info("Loaded %d papers from fixture", len(papers))
    else:
        papers = []
        async with OAIClient() as client:
            async for paper in client.harvest(from_date.date(), now.date()):
                papers.append(paper)
        log.info("Harvested %d papers from OAI-PMH", len(papers))

    # --- Chunk → embed → upsert (batched) ---
    last_pub = state.last_paper_published_at
    chunk_buf: list[Chunk] = []

    for paper in papers:
        chunks = chunk_paper(paper)
        chunk_buf.extend(chunks)
        if paper.published_at > last_pub:
            last_pub = paper.published_at

        if len(chunk_buf) >= _PIPELINE_BATCH:
            await _flush(chunk_buf, embedder, sparse_enc, store)
            chunk_buf.clear()

    if chunk_buf:
        await _flush(chunk_buf, embedder, sparse_enc, store)

    # --- Evict stale papers ---
    evicted = await store.evict_before(cutoff)
    if evicted:
        log.info("Evicted %d stale chunks", evicted)

    # --- Persist state ---
    new_state = IngestState(
        last_run_utc=now,
        last_oai_token=None,
        papers_in_index=await store.count(),
        last_paper_published_at=last_pub,
    )
    _save_state(new_state, state_path)
    log.info(
        "Ingest done: %d chunks in index, last paper %s",
        new_state.papers_in_index,
        new_state.last_paper_published_at.date(),
    )

    await store.close()
    return new_state


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="arXiv RAG ingestion pipeline")
    parser.add_argument("--days", type=int, default=90, metavar="N", help="Harvest window in days")
    parser.add_argument(
        "--evict-days", type=int, default=90, metavar="N", help="Evict papers older than N days"
    )
    parser.add_argument(
        "--fixture", action="store_true", help="Load from fixture file instead of OAI-PMH"
    )
    args = parser.parse_args()
    state = asyncio.run(
        run_ingest(days_back=args.days, evict_older_than_days=args.evict_days, fixture=args.fixture)
    )
    last = state.last_paper_published_at.date()
    print(f"papers_in_index={state.papers_in_index}  last_paper={last}")


if __name__ == "__main__":
    _cli()
