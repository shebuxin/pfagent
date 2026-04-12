"""Internal ANDES code-handling package.

Modules in this package hold pure functions extracted from
``src.chatbots.openai.rag_chatbot`` during the Stage 1 refactor
(see ``docs/REFACTOR_STAGE0_BASELINE.md``). The rag_chatbot module
re-exports every public name so external import paths stay stable.

Organization:

- ``detectors``  — intent/shape detection on user_context (pure, string-only)
- ``extractors`` — parse values out of user_context (IDs, buses, thresholds)

Additional modules (normalizer, validators, fallback, structured/*) are
added in subsequent Stage 1 batches.
"""
