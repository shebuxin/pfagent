"""Pure extractors that pull structured values out of user_context strings.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. Every
function here is side-effect free and depends only on ``re``. This is the
lowest layer of the andes_code package — ``detectors`` build on top of
this.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Matches a ```python ... ``` fenced block. Used by both the response
# extractor here and the code-block transformer in
# src.andes_code.normalizer. Keep this single source of truth so the
# two do not drift.
PYTHON_CODE_BLOCK_PATTERN: str = r"```python\s*\n(.*?)```"


def extract_python_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text"""
    matches = re.findall(PYTHON_CODE_BLOCK_PATTERN, text, re.DOTALL)
    return [code.strip() for code in matches]


def _extract_markdown_section(user_context: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^\s*## |\Z)",
        user_context or "",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return ""
    return match.group("body").strip()


def extract_effective_user_context(user_context: str) -> str:
    """Strip repo-aware repair prompts down to the user's actual intent plus continuity."""
    if not user_context:
        return ""

    original_request = _extract_markdown_section(user_context, "Original user request")
    if not original_request:
        return user_context

    sections = [original_request.strip()]

    continuity_context = _extract_markdown_section(user_context, "ANDES continuity context")
    if continuity_context:
        sections.append(f"ANDES continuity context:\n{continuity_context.strip()}")

    uploaded_case_preview = _extract_markdown_section(user_context, "Uploaded case preview")
    if uploaded_case_preview:
        sections.append(f"Uploaded case preview:\n{uploaded_case_preview.strip()}")

    runtime_file_context = _extract_markdown_section(user_context, "Runtime file context")
    if runtime_file_context and "files currently available" in runtime_file_context.lower():
        sections.append(runtime_file_context.strip())

    return "\n\n".join(section for section in sections if section.strip())


def extract_continuity_case_identifier(user_context: str) -> str:
    match = re.search(
        r"Last success(?:ful|fully executed) case identifier:\s*([^\n]+)",
        user_context or "",
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"Last successfully executed case identifier:\s*([^\n]+)",
            user_context or "",
            flags=re.IGNORECASE,
        )
    return match.group(1).strip() if match else ""


def extract_continuity_case_source(user_context: str) -> str:
    match = re.search(
        r"Last success(?:ful|fully executed) case source:\s*([^\n]+)",
        user_context or "",
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"Last successfully executed case source:\s*([^\n]+)",
            user_context or "",
            flags=re.IGNORECASE,
        )
    return match.group(1).strip().lower() if match else ""


def infer_requested_builtin_case(user_context: str) -> str:
    effective_user_context = extract_effective_user_context(user_context)
    normalized = (effective_user_context or "").lower()
    if (
        "pjm5bus" in normalized
        or "5bus" in normalized
        or "pjm 5-bus" in normalized
        or "pjm 5 bus" in normalized
        or "pjm5" in normalized
    ):
        return "5bus/pjm5bus.json"
    if "kundur_full" in normalized or "kundur" in normalized:
        return "kundur/kundur_full.xlsx"
    if "ieee 39" in normalized or "ieee39" in normalized:
        return "ieee39/ieee39.xlsx"
    if "ieee 14" in normalized or "ieee14" in normalized:
        return "ieee14/ieee14_full.xlsx"
    continuity_case = extract_continuity_case_identifier(effective_user_context or user_context)
    continuity_source = extract_continuity_case_source(effective_user_context or user_context)
    if continuity_case and (not continuity_source or continuity_source == "builtin"):
        return continuity_case.replace("\\", "/")
    return ""


def extract_requested_bus_number(user_context: str) -> str:
    match = re.search(r"\bbus\s+(\d+)\b", user_context or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def extract_requested_bus_numbers(user_context: str) -> List[str]:
    return re.findall(r"\bbus\s+(\d+)\b", user_context or "", flags=re.IGNORECASE)


def extract_uploaded_files_from_context(user_context: str) -> List[str]:
    """Extract uploaded filenames from runtime context injected by the app."""
    if not user_context:
        return []

    uploaded_files: List[str] = []
    in_uploaded_section = False
    for raw_line in user_context.splitlines():
        line = raw_line.strip()
        if "Uploaded files available during execution" in line:
            in_uploaded_section = True
            continue

        if not in_uploaded_section:
            continue

        if not line.startswith("- "):
            if line:
                in_uploaded_section = False
            continue

        candidate = line[2:].strip()
        lower_candidate = candidate.lower()
        if lower_candidate.startswith("use these filenames"):
            continue
        if lower_candidate.startswith("case-loading rule"):
            continue
        if lower_candidate.startswith("preferred uploaded-case template"):
            continue
        if "." not in candidate:
            continue
        uploaded_files.append(candidate)

    return uploaded_files


def _extract_voltage_bounds(user_context: str) -> Tuple[float, float]:
    match = re.search(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]", user_context)
    if not match:
        return 0.95, 1.05
    return float(match.group(1)), float(match.group(2))
