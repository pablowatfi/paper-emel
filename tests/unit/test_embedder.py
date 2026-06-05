"""Unit tests for BGEEmbedder — model is downloaded on first run (~30s)."""

from __future__ import annotations

import numpy as np
import pytest

from arxiv_rag.indexing.embedder import BGEEmbedder, Embedder

_TEXTS = [
    "Attention is all you need.",
    "Deep learning for natural language processing.",
    "Mixture of experts for large language models.",
]


class TestBGEEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self) -> BGEEmbedder:
        return BGEEmbedder()

    def test_shape(self, embedder: BGEEmbedder) -> None:
        vecs = embedder.embed_texts(_TEXTS)
        assert vecs.shape == (len(_TEXTS), 384)

    def test_dtype_is_float32(self, embedder: BGEEmbedder) -> None:
        vecs = embedder.embed_texts(_TEXTS)
        assert vecs.dtype == np.float32

    def test_determinism(self, embedder: BGEEmbedder) -> None:
        vecs1 = embedder.embed_texts(_TEXTS)
        vecs2 = embedder.embed_texts(_TEXTS)
        assert np.allclose(vecs1, vecs2, atol=1e-5)

    def test_l2_normalized(self, embedder: BGEEmbedder) -> None:
        vecs = embedder.embed_texts(_TEXTS)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_batch_of_200(self, embedder: BGEEmbedder) -> None:
        texts = [f"paper about topic number {i}" for i in range(200)]
        vecs = embedder.embed_texts(texts)
        assert vecs.shape == (200, 384)
        assert vecs.dtype == np.float32

    def test_dimensions_class_attribute(self) -> None:
        assert BGEEmbedder.dimensions == 384

    def test_satisfies_embedder_protocol(self, embedder: BGEEmbedder) -> None:
        e: Embedder = embedder
        assert e.dimensions == 384
        result = e.embed_texts(["hello"])
        assert result.shape == (1, 384)
