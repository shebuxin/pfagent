"""Smoke tests for src.chatbots.openai.embedding_config.

The dimensions here have to match OpenAI's actual embedding model
dimensions or the FAISS IndexFlatL2 initialization in RAGChatbot will
misalign with every vector shipped to/from the store. Pin the values.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.chatbots.openai.embedding_config import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_DIMENSIONS,
    resolve_embedding_dimension,
)


class EmbeddingConfigTests(unittest.TestCase):
    def test_known_models_have_expected_dimensions(self):
        # These have to match OpenAI. If they drift, FAISS index and
        # embedding vectors will no longer agree on shape.
        self.assertEqual(EMBEDDING_MODEL_DIMENSIONS["text-embedding-3-small"], 1536)
        self.assertEqual(EMBEDDING_MODEL_DIMENSIONS["text-embedding-3-large"], 3072)
        self.assertEqual(EMBEDDING_MODEL_DIMENSIONS["text-embedding-ada-002"], 1536)

    def test_resolve_returns_registered_dimension(self):
        self.assertEqual(resolve_embedding_dimension("text-embedding-3-large"), 3072)
        self.assertEqual(resolve_embedding_dimension("text-embedding-3-small"), 1536)

    def test_unknown_model_falls_back_to_default(self):
        self.assertEqual(
            resolve_embedding_dimension("text-embedding-future-9000"),
            DEFAULT_EMBEDDING_DIMENSION,
        )

    def test_default_dimension_is_sane(self):
        # A non-positive default would break FAISS at index creation.
        self.assertGreater(DEFAULT_EMBEDDING_DIMENSION, 0)

    def test_rag_chatbot_still_reexports_symbols(self):
        # External callers (if any) that still import from the old
        # location must keep working after the move.
        from src.chatbots.openai import rag_chatbot
        self.assertIs(rag_chatbot.resolve_embedding_dimension, resolve_embedding_dimension)
        self.assertIs(rag_chatbot.EMBEDDING_MODEL_DIMENSIONS, EMBEDDING_MODEL_DIMENSIONS)


if __name__ == "__main__":
    unittest.main()
