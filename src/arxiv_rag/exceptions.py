class ArxivRagError(Exception):
    """Base exception for all arxiv-rag errors."""


class IngestionError(ArxivRagError):
    """Raised when the ingestion pipeline fails unrecoverably."""


class RetrievalError(ArxivRagError):
    """Raised when retrieval from the vector store fails."""


class GenerationError(ArxivRagError):
    """Raised when all LLM providers fail and degraded mode is not acceptable."""


class EvalError(ArxivRagError):
    """Raised when the evaluation pipeline encounters an unrecoverable error."""
