"""Dense embedding via BGE-small-en-v1.5."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Structural interface for text embedding models."""

    dimensions: int

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


class BGEEmbedder:
    """BGE-small-en-v1.5 embedder with lazy model load and L2 normalisation."""

    dimensions: int = 384
    _MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    _BATCH_SIZE: int = 64

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer as _ST

            # Force CPU: avoids MPS/CUDA OOM on machines where the GPU is present
            # but constrained (e.g. Mac with shared MPS memory).  BGE-small is
            # fast enough on CPU for our batch sizes.
            self._model = _ST(self._MODEL_NAME, device="cpu")
        return self._model

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed texts; returns (n_texts, 384) float32 array, L2-normalised."""
        vecs = self._load().encode(
            texts,
            batch_size=self._BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)
