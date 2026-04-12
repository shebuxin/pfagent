"""Structured deterministic ANDES code generation.

This subpackage holds the structured codegen pipeline extracted from
``src.chatbots.openai.rag_chatbot`` in Stage 1:

- ``state``   — ``StructuredAndesState`` dataclass + parsers/extractors
                that derive state from user_context, plus the report-kind
                inference and applicability gates.
- ``scripts`` — template-based script builders, one per report kind.
- ``codegen`` — the top-level ``build_structured_andes_response``
                orchestrator.
"""
