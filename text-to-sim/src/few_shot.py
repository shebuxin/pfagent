import json
import os
from typing import Dict, List, Any
from src.andes_case_catalog import build_andes_builtin_cases_section


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return max(parsed, 0)
    except ValueError:
        return default


def load_andes_few_shot_examples(path: str = None) -> List[Dict[str, str]]:
    """Load ANDES few-shot examples from JSON file."""
    if not _env_flag("ENABLE_ANDES_FEW_SHOT", True):
        return []

    default_path = os.path.join("data_files", "few_shot_andes.json")
    file_path = path or os.environ.get("ANDES_FEW_SHOT_PATH", default_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload: Any = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("examples", [])
    else:
        return []

    examples: List[Dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        user = item.get("user")
        assistant = item.get("assistant")
        if isinstance(user, str) and user.strip() and isinstance(assistant, str) and assistant.strip():
            examples.append(
                {
                    "id": str(item.get("id", "")),
                    "user": user.strip(),
                    "assistant": assistant.strip(),
                }
            )
    return examples


def build_andes_few_shot_section(max_examples: int = None) -> str:
    """Build a compact few-shot text section for system prompt injection."""
    examples = load_andes_few_shot_examples()
    cases_section = build_andes_builtin_cases_section(limit=_env_int("ANDES_CASE_LIST_LIMIT", 40))
    if not examples and not cases_section:
        return ""

    limit = max_examples if max_examples is not None else _env_int("ANDES_FEW_SHOT_LIMIT", 4)
    if limit == 0:
        return ""

    selected = examples[:limit] if examples else []
    lines: List[str] = []
    if selected:
        lines.append("ANDES Code Generation Few-shot Examples:")
        lines.append("- Follow the style and constraints demonstrated below.")
        for idx, example in enumerate(selected, 1):
            lines.append("")
            lines.append(f"Example {idx}")
            lines.append(f"User: {example['user']}")
            lines.append("Assistant:")
            lines.append(example["assistant"])

    if cases_section:
        if lines:
            lines.append("")
        lines.append(cases_section)
    return "\n".join(lines)
