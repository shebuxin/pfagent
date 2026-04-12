"""Pure predicates that classify intent/shape of user_context strings.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. These
functions take a ``user_context`` (and sometimes a response body) and
return bool or other primitive types. They depend only on ``re`` and on
``src.andes_code.extractors``.
"""

from __future__ import annotations

import re

from src.andes_code.extractors import extract_effective_user_context


def is_code_only_request(user_context: str) -> bool:
    normalized_user_context = (user_context or "").lower()
    if "runnable python code only" in normalized_user_context or "code only" in normalized_user_context:
        return True

    patterns = (
        r"exactly one runnable python code block",
        r"one complete runnable python script only",
        r"one runnable python script only",
        r"one full python script",
        r"one full script only",
        r"one new complete script only",
        r"return a fresh complete script",
        r"single\s+```python",
        r"single\s+`python block",
        r"no prose",
        r"nothing else",
    )
    return any(re.search(pattern, normalized_user_context) for pattern in patterns)


def is_explanatory_followup_request(user_context: str) -> bool:
    """Detect conceptual/debugging follow-ups that should be answered in prose."""
    normalized_user_context = (extract_effective_user_context(user_context) or user_context or "").strip().lower()
    if not normalized_user_context:
        return False

    explicit_code_markers = (
        "runnable python",
        "python code",
        "python script",
        "code only",
        "```python",
        "result_json",
        "return json",
        "write code",
        "generate code",
    )
    if any(marker in normalized_user_context for marker in explicit_code_markers):
        return False

    execution_markers = (
        "run a power flow",
        "run power flow",
        "rerun",
        "plot ",
        "save ",
        "load the case",
        "load case",
        "use ieee",
        "use kundur",
        "use the built-in",
        "trip one line",
        "trip a line",
        "trip the line",
        "open one line",
        "open the line",
        "disconnect the line",
        "line outage",
        "n-1",
        "modify",
        "set the ",
        "scale ",
        "increase ",
        "decrease ",
        "add one",
        "create a",
        "generate a plot",
    )
    if any(marker in normalized_user_context for marker in execution_markers):
        return False

    explanation_patterns = (
        r"^\s*why\b",
        r"^\s*explain\b",
        r"\bcan you explain\b",
        r"\bwhat happened\b",
        r"\bwhat does .+\bmean\b",
        r"\bdoes that mean\b",
        r"\bis that expected\b",
        r"\bhow should i interpret\b",
        r"\bhow do i interpret\b",
        r"\bwhy does\b",
        r"\bwhy did\b",
        r"\bwhy doesn't\b",
        r"\bwhy does not\b",
    )
    return any(re.search(pattern, normalized_user_context) for pattern in explanation_patterns)


def looks_like_python_script(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("```"):
        return False

    signals = (
        "import ",
        "from ",
        "print(",
        "andes.",
        "ssa =",
        "case =",
        "plt.",
        "np.",
        "os.path.join",
    )
    hit_count = sum(1 for signal in signals if signal in stripped)
    return hit_count >= 2 or ("import " in stripped and "\n" in stripped)


def prompt_explicitly_mentions_idx(user_context: str) -> bool:
    return bool(re.search(r"\bidx\b\s*(?:=|is|to|['\"])", user_context or "", flags=re.IGNORECASE))
