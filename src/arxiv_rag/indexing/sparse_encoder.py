"""BM25-style sparse encoder producing Qdrant SparseVectors.

Uses CRC32 hashing for stable term IDs across processes (avoids Python's
randomised hash seed). Vocab size 30 000 gives negligible collision rate
for short abstract-length texts.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

_VOCAB_SIZE = 30_000


def _term_id(token: str) -> int:
    return zlib.crc32(token.encode()) % _VOCAB_SIZE


def _encode(text: str) -> SparseVector:
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(_term_id(t) for t in tokens)
    total = sum(counts.values())
    pairs = sorted(counts.items())
    return SparseVector(
        indices=[idx for idx, _ in pairs],
        values=[cnt / total for _, cnt in pairs],
    )


class SparseEncoder:
    """Encodes a batch of texts into Qdrant SparseVectors."""

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        return [_encode(t) for t in texts]
