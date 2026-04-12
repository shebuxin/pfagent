from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


_FILE_REFERENCE_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:xlsx|xls|json|csv)\b",
    re.IGNORECASE,
)

_GET_CASE_PATTERN = re.compile(
    r"andes\.get_case\(\s*[\"']([^\"']+)[\"']\s*\)",
    re.IGNORECASE,
)

_BUS_REFERENCE_PATTERN = re.compile(r"\bbus(?:es)?\s+\d+\b", re.IGNORECASE)
_LINE_ENDPOINT_PATTERN = re.compile(r"\bline\b|\bbus1\b|\bbus2\b", re.IGNORECASE)


@dataclass
class ConversationCompactionConfig:
    enabled: bool = True
    trigger_message_count: int = 14
    keep_recent_messages: int = 8
    max_summary_chars: int = 3200
    max_objective_bullets: int = 4
    max_recent_bullets: int = 8
    max_reference_items: int = 8
    readable_recent_messages: int = 6


def build_compacted_message_window(
    system_message: str,
    conversation_history: Sequence[BaseMessage],
    config: ConversationCompactionConfig,
) -> Tuple[List[BaseMessage], str]:
    history = list(conversation_history)
    if not history:
        return [SystemMessage(content=system_message)], ""

    if not config.enabled or len(history) <= config.trigger_message_count:
        return [SystemMessage(content=system_message)] + history, ""

    keep_recent = max(1, min(config.keep_recent_messages, len(history)))
    older_messages = history[:-keep_recent]
    recent_messages = history[-keep_recent:]
    summary_body = build_compaction_summary(older_messages, config)
    if not summary_body:
        return [SystemMessage(content=system_message)] + history, ""

    summary_message = SystemMessage(
        content=(
            "Continuation Memory:\n"
            "You are continuing an existing conversation. The summary below compresses older turns.\n"
            "Treat it as authoritative session memory, preserve the established case/file references,\n"
            "device edits, output requirements, and user preferences, and continue directly without\n"
            "asking the user to repeat earlier details unless necessary.\n\n"
            f"{summary_body}"
        )
    )
    return [SystemMessage(content=system_message), summary_message] + recent_messages, summary_body


def build_readable_conversation_summary(
    conversation_history: Sequence[BaseMessage],
    config: ConversationCompactionConfig,
) -> str:
    history = list(conversation_history)
    if not history:
        return "No conversation history available."

    sections: List[str] = []
    if config.enabled and len(history) > config.trigger_message_count:
        keep_recent = max(1, min(config.keep_recent_messages, len(history)))
        older_messages = history[:-keep_recent]
        summary_body = build_compaction_summary(older_messages, config)
        if summary_body:
            sections.append(f"Compacted earlier session:\n{summary_body}")

    recent_limit = max(1, config.readable_recent_messages)
    recent_messages = history[-recent_limit:]
    recent_lines = [
        f"{_message_role(message)}: {_summarize_message(message, max_chars=140)}"
        for message in recent_messages
        if _summarize_message(message, max_chars=140)
    ]
    if recent_lines:
        sections.append("Recent live messages:\n" + "\n".join(recent_lines))

    return "\n\n".join(section for section in sections if section).strip() or "No conversation history available."


def build_compaction_summary(
    conversation_history: Sequence[BaseMessage],
    config: ConversationCompactionConfig,
) -> str:
    history = [message for message in conversation_history if _message_text(message).strip()]
    if not history:
        return ""

    objectives = _collect_user_objectives(history, config.max_objective_bullets)
    references = _collect_references(history, config.max_reference_items)
    constraints = _collect_constraints(history)
    recent_bullets = _build_recent_turn_bullets(history, config.max_recent_bullets)

    sections: List[str] = []
    if objectives:
        sections.append("Active user goals carried over:\n" + "\n".join(f"- {item}" for item in objectives))
    if references:
        sections.append("Established case and device references:\n" + "\n".join(f"- {item}" for item in references))
    if constraints:
        sections.append("Carry forward these constraints:\n" + "\n".join(f"- {item}" for item in constraints))
    if recent_bullets:
        sections.append("Condensed earlier turns:\n" + "\n".join(recent_bullets))

    summary = "\n\n".join(section for section in sections if section).strip()
    if len(summary) <= config.max_summary_chars:
        return summary

    trimmed_recent = list(recent_bullets)
    while trimmed_recent and len(summary) > config.max_summary_chars:
        trimmed_recent.pop(0)
        sections = []
        if objectives:
            sections.append("Active user goals carried over:\n" + "\n".join(f"- {item}" for item in objectives))
        if references:
            sections.append("Established case and device references:\n" + "\n".join(f"- {item}" for item in references))
        if constraints:
            sections.append("Carry forward these constraints:\n" + "\n".join(f"- {item}" for item in constraints))
        if trimmed_recent:
            sections.append("Condensed earlier turns:\n" + "\n".join(trimmed_recent))
        summary = "\n\n".join(section for section in sections if section).strip()

    if len(summary) <= config.max_summary_chars:
        return summary

    return summary[: max(config.max_summary_chars - 3, 0)].rstrip() + "..."


def _collect_user_objectives(conversation_history: Sequence[BaseMessage], max_items: int) -> List[str]:
    objectives: List[str] = []
    for message in conversation_history:
        if not isinstance(message, HumanMessage):
            continue
        text = _message_text(message)
        if _is_retry_feedback(text):
            continue
        summary = _truncate_text(text, max_chars=220)
        if not summary:
            continue
        objectives.append(summary)
    return _dedupe_keep_last(objectives, max_items)


def _collect_references(conversation_history: Sequence[BaseMessage], max_items: int) -> List[str]:
    files = set()
    case_paths = set()
    buses = set()
    line_related = False

    for message in conversation_history:
        text = _message_text(message)
        for case_path in _GET_CASE_PATTERN.findall(text):
            case_paths.add(case_path)
        for filename in _FILE_REFERENCE_PATTERN.findall(text):
            files.add(filename)
        for bus_match in _BUS_REFERENCE_PATTERN.findall(text):
            buses.add(bus_match.lower())
        if _LINE_ENDPOINT_PATTERN.search(text):
            line_related = True

    references: List[str] = []
    if case_paths:
        references.append("Built-in case paths already referenced: " + ", ".join(sorted(case_paths)[:max_items]))
    if files:
        references.append("Files already referenced: " + ", ".join(sorted(files)[:max_items]))
    if buses:
        references.append("Bus targets mentioned earlier: " + ", ".join(sorted(buses)[:max_items]))
    if line_related:
        references.append("Line-based device edits or line analysis were already part of the session.")
    return references[:max_items]


def _collect_constraints(conversation_history: Sequence[BaseMessage]) -> List[str]:
    combined = "\n".join(_message_text(message) for message in conversation_history).lower()
    constraints: List[str] = []

    if "runnable python code only" in combined or "code only" in combined:
        constraints.append("Return one runnable Python script only.")
    if "no prose" in combined or "nothing else" in combined:
        constraints.append("Do not add prose outside the code block.")
    if "result_json" in combined:
        constraints.append("Preserve RESULT_JSON output requirements when requested.")
    if "exact uploaded filename" in combined or "uploaded-case template" in combined:
        constraints.append("Keep using the exact uploaded filename; do not rename uploaded cases.")
    if (
        "resolve the real `idx`" in combined
        or "resolve the real idx" in combined
        or "resolve idx from the case" in combined
        or "never guess internal `idx`" in combined
        or "do not guess idx" in combined
        or "don't guess idx" in combined
    ):
        constraints.append("Resolve real device idx values from the case before calling .set(...).")
    if "before setup" in combined:
        constraints.append("Apply pre-setup edits before calling setup when the workflow requires it.")

    return constraints


def _build_recent_turn_bullets(conversation_history: Sequence[BaseMessage], max_items: int) -> List[str]:
    summaries: List[str] = []
    for message in conversation_history:
        summary = _summarize_message(message, max_chars=180)
        if not summary:
            continue
        summaries.append(f"- {_message_role(message)}: {summary}")
    return summaries[-max_items:]


def _summarize_message(message: BaseMessage, max_chars: int = 180) -> str:
    text = _message_text(message)
    if not text:
        return ""

    if isinstance(message, HumanMessage) and _is_retry_feedback(text):
        return "Requested a corrected runnable script after validation or compilation feedback."

    if isinstance(message, AIMessage):
        summary = _summarize_ai_message(text)
        if summary:
            return _truncate_text(summary, max_chars=max_chars)

    if isinstance(message, ToolMessage):
        return _truncate_text(text, max_chars=max_chars)

    return _truncate_text(text, max_chars=max_chars)


def _summarize_ai_message(text: str) -> str:
    normalized = " ".join(text.split())
    if "```python" not in normalized and "import " not in normalized:
        return normalized

    behaviors: List[str] = []
    case_refs = _GET_CASE_PATTERN.findall(text)
    file_refs = _FILE_REFERENCE_PATTERN.findall(text)

    if "andes.load" in text or "andes.get_case" in text:
        behaviors.append("prepared ANDES case-loading code")
    if '.add("' in text or ".add(" in text:
        behaviors.append("adds or edits devices before setup")
    if ".set(" in text:
        behaviors.append("modifies existing device parameters")
    if "PFlow.run" in text:
        behaviors.append("runs power flow")
    if "plt." in text:
        behaviors.append("creates a plot artifact")
    if "RESULT_JSON" in text:
        behaviors.append("prints RESULT_JSON for machine checking")

    if behaviors:
        suffix_parts: List[str] = []
        if case_refs:
            suffix_parts.append("built-in case(s): " + ", ".join(sorted(set(case_refs))[:4]))
        if file_refs:
            suffix_parts.append("file(s): " + ", ".join(sorted(set(file_refs))[:4]))
        suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
        return "Generated runnable ANDES code that " + ", ".join(behaviors) + suffix + "."

    return normalized


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, AIMessage):
        return "Assistant"
    if isinstance(message, ToolMessage):
        return "Tool"
    if isinstance(message, SystemMessage):
        return "System"
    return message.__class__.__name__


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _dedupe_keep_last(items: Sequence[str], max_items: int) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in reversed(items):
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= max_items:
            break
    return list(reversed(result))


def _is_retry_feedback(text: str) -> bool:
    normalized = (text or "").lower()
    retry_signals = (
        "the python code in your previous response has",
        "please fix these errors",
        "please fix these issues",
        "compilation errors",
        "andes-specific issues",
        "provide a corrected response",
        "provide one corrected runnable python script only",
    )
    return any(signal in normalized for signal in retry_signals)
