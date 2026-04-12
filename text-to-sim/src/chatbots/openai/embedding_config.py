"""OpenAI embedding-model configuration data.

Extracted from ``rag_chatbot.py`` in Stage 3 so that operators who want
to add a new embedding model (or change the default fallback) only need
to touch one focused file instead of the 800-line chatbot module.
"""

from __future__ import annotations


# Embedding dimensions for supported OpenAI embedding models.
# See https://platform.openai.com/docs/guides/embeddings for the
# authoritative list; keep this dict in sync when OpenAI ships new
# embedding models that the app needs to serve.
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# Dimension returned when an unknown model name is supplied. Chosen to
# match ``text-embedding-3-small`` / ``text-embedding-ada-002`` because
# that is what most downstream callers assume when pressed.
DEFAULT_EMBEDDING_DIMENSION: int = 1536


def resolve_embedding_dimension(model_name: str) -> int:
    """Return the vector dimension for a given OpenAI embedding model.

    Unknown model names fall back to ``DEFAULT_EMBEDDING_DIMENSION``
    rather than raising, so a new model shipped by OpenAI still works
    out of the box (at the cost of possibly wrong dimensions; update
    ``EMBEDDING_MODEL_DIMENSIONS`` when that happens).
    """
    return EMBEDDING_MODEL_DIMENSIONS.get(model_name, DEFAULT_EMBEDDING_DIMENSION)
