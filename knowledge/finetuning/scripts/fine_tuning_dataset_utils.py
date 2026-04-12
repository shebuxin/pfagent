import ast
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Path layout after the 2026-04-10 knowledge/ refactor:
#   knowledge/finetuning/scripts/ -- this file
#   knowledge/finetuning/data/    -- jsonl + json datasets
#   knowledge/finetuning/generated/current/ -- generated prompts CSVs
#   knowledge/rag/code_examples/  -- few-shot / RAG code snippets
#   <repo_root>/text-to-sim/      -- app runtime
SCRIPTS_DIR = Path(__file__).resolve().parent
FINETUNING_ROOT = SCRIPTS_DIR.parent
DATA_DIR = FINETUNING_ROOT / "data"
GENERATED_DIR = FINETUNING_ROOT / "generated" / "current"
KNOWLEDGE_ROOT = FINETUNING_ROOT.parent
REPO_ROOT = KNOWLEDGE_ROOT.parent
TEXT_TO_SIM_ROOT = REPO_ROOT / "text-to-sim"

# Retained name for backwards-compatible call sites inside this package.
ROOT = SCRIPTS_DIR

DEFAULT_CODE_DIR = KNOWLEDGE_ROOT / "rag" / "code_examples"
DEFAULT_CANONICAL_TASKS_CSV = DATA_DIR / "examples.csv"
DEFAULT_LOW_LEVEL_PROMPTS_CSV = GENERATED_DIR / "low_level_generated_prompts.csv"
DEFAULT_HIGH_LEVEL_PROMPTS_CSV = GENERATED_DIR / "high_level_generated_prompts.csv"
DEFAULT_SUMMARY_CSV = GENERATED_DIR / "generated_output_summary.csv"
DEFAULT_FEW_SHOT_JSON = TEXT_TO_SIM_ROOT / "data_files" / "few_shot_andes.json"
DEFAULT_CURATED_EXAMPLES_JSON = DATA_DIR / "curated_training_examples.json"
DEFAULT_VERIFIED_EXAMPLES_JSON = DATA_DIR / "verified_training_examples.json"
DEFAULT_GENERALIZED_VERIFIED_EXAMPLES_JSON = DATA_DIR / "generalized_verified_training_examples.json"
DEFAULT_RAW_JSONL = DATA_DIR / "fine_tuning_data.jsonl"
DEFAULT_CLEAN_JSONL = DATA_DIR / "fine_tuning_data.cleaned.jsonl"
DEFAULT_AUDIT_JSON = DATA_DIR / "fine_tuning_data.audit.json"

PROMPT_KEYWORDS = (
    "andes",
    "power flow",
    "power-flow",
    "grid",
    "system",
    "generator",
    "voltage",
    "angle",
    "bus",
    "line",
    "load",
    "slack",
    "pv",
    "pq",
    "case",
)

BANNED_PROMPT_MARKERS = (
    "code output:",
    "return code:",
    "traceback",
    "stderr:",
    "stdout:",
)

BANNED_ASSISTANT_MARKERS = (
    "code output:",
    "return code:",
    "traceback",
    "the provided script is intended",
    "the supplied script is intended",
    "those outputs would then let you inspect",
)

LEGACY_CASE_REWRITES = {
    "ieee39_base.xlsx": "ieee39/ieee39.xlsx",
    "EI_33.xlsx": "ei/EI_33.xlsx",
}


@dataclass
class DatasetPair:
    prompt: str = ""
    assistant: str = ""
    source: str = ""
    code_file: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


def normalize_common_text(text: str) -> str:
    normalized = (text or "").replace("\u00a0", " ")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\t", "    ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized


def normalize_prose_text(text: str) -> str:
    normalized = normalize_common_text(text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_code_text(text: str) -> str:
    normalized = normalize_common_text(text)
    lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_prompt(prompt: str) -> str:
    prompt = normalize_prose_text(prompt)
    return re.sub(r"\s+", " ", prompt).strip()


def strip_code_fences(code: str) -> str:
    fenced = normalize_code_text(code)
    match = re.fullmatch(r"```(?:python)?\s*\n(.*)\n```", fenced, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return fenced


def remove_trailing_task_comments(code: str) -> str:
    lines = code.rstrip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    trailing_comment_block: List[str] = []
    index = len(lines) - 1
    while index >= 0 and lines[index].lstrip().startswith("#"):
        trailing_comment_block.append(lines[index])
        index -= 1

    if trailing_comment_block and any(len(line.lstrip("# ").strip()) > 30 for line in trailing_comment_block):
        lines = lines[: index + 1]
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines).strip()


def remove_argument_from_andes_load(code: str, argument_name: str) -> str:
    updated = re.sub(rf",\s*{argument_name}\s*=\s*False", "", code)
    updated = re.sub(rf"{argument_name}\s*=\s*False\s*,\s*", "", updated)
    return updated


def rewrite_legacy_case_assignments(code: str) -> str:
    rewritten = code
    for old_name, new_path in LEGACY_CASE_REWRITES.items():
        rewritten = re.sub(
            rf"(?m)^(?P<lhs>\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)os\.path\.join\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*['\"]{re.escape(old_name)}['\"]\s*\)\s*$",
            rf"\g<lhs>andes.get_case('{new_path}')",
            rewritten,
        )
        rewritten = re.sub(
            rf"(?m)^(?P<lhs>\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)['\"]{re.escape(old_name)}['\"]\s*$",
            rf"\g<lhs>andes.get_case('{new_path}')",
            rewritten,
        )
        rewritten = rewritten.replace(f"'{old_name}'", f"'andes.get_case(\"{new_path}\")'")
    return rewritten


def cleanup_stringified_get_case(code: str) -> str:
    updated = code
    updated = updated.replace("'andes.get_case(\"", "andes.get_case(\"")
    updated = updated.replace("\")'", "\")")
    return updated


def rewrite_bus_uid_patterns(code: str) -> str:
    updated = code

    updated = re.sub(
        r"bus_idx = ssa\.Bus\.idx\.v\s*\nidx_max_sheet = bus_idx\[idx_max\]\s*\n\s*\n# Retrieve the bus ID from BUS sheet\s*\nbus_ids = ssa\.Bus\.uid\s*\nmax_bus = bus_ids\[idx_max_sheet\]",
        "bus_idx = ssa.Bus.idx.v\nmax_bus = bus_idx[idx_max]",
        updated,
    )
    updated = re.sub(
        r"bus_idx = ssa\.Bus\.idx\.v\s*# mapping from position -> sheet index\s*\nbus_ids = ssa\.Bus\.uid\s*# array of bus UIDs keyed by sheet index\s*\nsheet_max = bus_idx\[idx_max\]\s*\nsheet_min = bus_idx\[idx_min\]\s*\nbus_max = bus_ids\[sheet_max\]\s*\nbus_min = bus_ids\[sheet_min\]",
        "bus_idx = ssa.Bus.idx.v\nbus_max = bus_idx[idx_max]\nbus_min = bus_idx[idx_min]",
        updated,
    )
    updated = updated.replace("Bus UID", "Bus")
    updated = updated.replace("ssa.Bus.uid", "ssa.Bus.idx.v")
    return updated


def rewrite_add_model_keyword(code: str) -> str:
    return re.sub(
        r"(\.add\()\s*model\s*=\s*([\"'][^\"']+[\"'])\s*,\s*",
        r"\1\2, ",
        code,
    )


def drop_unused_script_dir_and_os_import(code: str) -> str:
    updated = code
    if "script_dir" in updated and "os.getcwd()" in updated and "script_dir" not in updated.replace("script_dir = os.getcwd()", ""):
        updated = re.sub(r"(?m)^script_dir = os\.getcwd\(\)\n?", "", updated)
    if "os.path.join" not in updated and "os.getcwd()" not in updated:
        updated = re.sub(r"(?m)^import os\n", "", updated)
    return updated


def sanitize_assistant_code(code: str) -> str:
    sanitized = strip_code_fences(code)
    sanitized = normalize_code_text(sanitized)
    sanitized = remove_trailing_task_comments(sanitized)
    sanitized = remove_argument_from_andes_load(sanitized, "default_config")
    sanitized = rewrite_add_model_keyword(sanitized)
    sanitized = rewrite_legacy_case_assignments(sanitized)
    sanitized = cleanup_stringified_get_case(sanitized)
    sanitized = rewrite_bus_uid_patterns(sanitized)
    sanitized = drop_unused_script_dir_and_os_import(sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    return sanitized


def normalize_message_content(role: str, content: str) -> str:
    if role == "assistant":
        return sanitize_assistant_code(content)
    if role == "user":
        return normalize_prompt(content)
    return normalize_prose_text(content)


def normalize_messages(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized_messages: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if not role:
            continue
        normalized_messages.append({"role": role, "content": normalize_message_content(role, content)})
    return normalized_messages


def prompt_is_specific_enough(prompt: str) -> bool:
    normalized = normalize_prompt(prompt).lower()
    if len(normalized.split()) < 4:
        return False
    if any(marker in normalized for marker in BANNED_PROMPT_MARKERS):
        return False
    return any(keyword in normalized for keyword in PROMPT_KEYWORDS)


def assistant_is_obviously_bad(assistant: str) -> Optional[str]:
    normalized = normalize_code_text(assistant)
    lowered = normalized.lower()

    for marker in BANNED_ASSISTANT_MARKERS:
        if marker in lowered:
            return f"banned_assistant_marker:{marker}"

    banned_code_markers = (
        "ieee39_base.xlsx",
        "Bus.uid",
        "Code Output:",
        "Return Code:",
    )
    for marker in banned_code_markers:
        if marker in normalized:
            return f"banned_code_marker:{marker}"

    if not ("import andes" in normalized or "andes.load(" in normalized):
        return "missing_andes_usage"

    try:
        ast.parse(normalized)
    except SyntaxError as exc:
        return f"syntax_error:{exc.msg}"

    return None


def load_code_examples(code_dir: Path = DEFAULT_CODE_DIR) -> Dict[str, str]:
    code_map: Dict[str, str] = {}
    for code_path in sorted(code_dir.glob("*.py")):
        code_map[code_path.name] = sanitize_assistant_code(code_path.read_text(encoding="utf-8"))
    return code_map


def load_canonical_pairs(code_map: Dict[str, str], examples_csv: Path = DEFAULT_CANONICAL_TASKS_CSV) -> List[DatasetPair]:
    pairs: List[DatasetPair] = []
    with examples_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code_file = row.get("Code File", "").strip()
            prompt = row.get("Task", "").strip()
            assistant = code_map.get(code_file)
            if code_file and prompt and assistant:
                pairs.append(DatasetPair(prompt=prompt, assistant=assistant, source="canonical_task", code_file=code_file))
    return pairs


def load_prompt_csv_pairs(
    csv_path: Path,
    source_name: str,
    code_map: Dict[str, str],
    require_specific_prompts: bool = True,
) -> List[DatasetPair]:
    if not csv_path.exists():
        return []

    pairs: List[DatasetPair] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row.get("Question", "").strip()
            code_file = row.get("File Path to Answer", "").strip()
            assistant = code_map.get(code_file)
            if not prompt or not code_file or not assistant:
                continue
            if require_specific_prompts and not prompt_is_specific_enough(prompt):
                continue
            pairs.append(DatasetPair(prompt=prompt, assistant=assistant, source=source_name, code_file=code_file))
    return pairs


def load_structured_pairs(json_path: Path, source_name: str) -> List[DatasetPair]:
    if not json_path.exists():
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    pairs: List[DatasetPair] = []
    for item in examples:
        if isinstance(item.get("messages"), list):
            messages = normalize_messages(item.get("messages", []))
            if messages:
                pairs.append(
                    DatasetPair(
                        source=source_name,
                        code_file=item.get("id"),
                        messages=messages,
                    )
                )
            continue
        prompt = item.get("user", "").strip()
        assistant = sanitize_assistant_code(item.get("assistant", ""))
        if prompt and assistant:
            pairs.append(DatasetPair(prompt=prompt, assistant=assistant, source=source_name, code_file=item.get("id")))
    return pairs


def load_few_shot_pairs(few_shot_json: Path = DEFAULT_FEW_SHOT_JSON) -> List[DatasetPair]:
    return load_structured_pairs(few_shot_json, "few_shot")


def load_curated_pairs(curated_json: Path = DEFAULT_CURATED_EXAMPLES_JSON) -> List[DatasetPair]:
    return load_structured_pairs(curated_json, "curated_hard_case")


def load_verified_pairs(verified_json: Path = DEFAULT_VERIFIED_EXAMPLES_JSON) -> List[DatasetPair]:
    return load_structured_pairs(verified_json, "strict_verified_scenario")


def load_generalized_verified_pairs(
    verified_json: Path = DEFAULT_GENERALIZED_VERIFIED_EXAMPLES_JSON,
) -> List[DatasetPair]:
    return load_structured_pairs(verified_json, "generalized_verified_conversation")


def validate_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    if len(messages) < 2:
        return "too_few_messages"
    if messages[-1].get("role") != "assistant":
        return "conversation_must_end_with_assistant"

    assistant_count = 0
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "")
        content = str(message.get("content", ""))
        lowered = content.lower()

        if role not in {"system", "user", "assistant"}:
            return f"invalid_role:{role or 'missing'}"
        if not content:
            return f"empty_{role}_message"

        if role == "user":
            if any(marker in lowered for marker in BANNED_PROMPT_MARKERS):
                return f"user_message_{index}:banned_prompt_marker"
        elif role == "assistant":
            assistant_count += 1
            reason = assistant_is_obviously_bad(content)
            if reason:
                return f"assistant_message_{index}:{reason}"

    if assistant_count == 0:
        return "missing_assistant_message"
    return None


def dedupe_pairs(pairs: Iterable[DatasetPair]) -> Tuple[List[DatasetPair], int]:
    unique_pairs: List[DatasetPair] = []
    seen = set()
    duplicate_count = 0
    for pair in pairs:
        if pair.messages:
            key = tuple(
                (
                    message["role"],
                    re.sub(r"\s+", " ", message["content"].strip()),
                )
                for message in pair.messages
            )
        else:
            key = (
                re.sub(r"\s+", " ", pair.prompt.strip().lower()),
                re.sub(r"\s+", " ", pair.assistant.strip()),
            )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_pairs.append(pair)
    return unique_pairs, duplicate_count


def audit_jsonl(jsonl_path: Path) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    examples: Dict[str, Dict[str, str]] = {}

    if not jsonl_path.exists():
        return {"path": str(jsonl_path), "exists": False, "counts": {}, "examples": {}}

    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            counts["invalid_json"] += 1
            continue

        messages = obj.get("messages", [])
        if len(messages) != 2:
            counts["bad_message_count"] += 1
            continue

        prompt = messages[0].get("content", "")
        assistant = messages[1].get("content", "")

        if "ieee39_base.xlsx" in assistant:
            counts["legacy_ieee39_base"] += 1
            examples.setdefault("legacy_ieee39_base", {"line": str(line_number), "prompt": prompt[:160], "assistant": assistant[:220]})
        if "Bus.uid" in assistant:
            counts["bus_uid_usage"] += 1
            examples.setdefault("bus_uid_usage", {"line": str(line_number), "prompt": prompt[:160], "assistant": assistant[:220]})
        if "default_config=False" in assistant:
            counts["default_config_false"] += 1
        if "Code Output:" in prompt or "Return Code:" in prompt:
            counts["execution_trace_in_prompt"] += 1
            examples.setdefault("execution_trace_in_prompt", {"line": str(line_number), "prompt": prompt[:220], "assistant": assistant[:160]})
        if any(marker in assistant.lower() for marker in BANNED_ASSISTANT_MARKERS):
            counts["narrative_summary_answer"] += 1
            examples.setdefault("narrative_summary_answer", {"line": str(line_number), "prompt": prompt[:160], "assistant": assistant[:220]})

    return {
        "path": str(jsonl_path),
        "exists": True,
        "counts": dict(counts),
        "examples": examples,
    }


def build_clean_dataset(
    include_high_level: bool = False,
    include_generated_summaries: bool = False,
) -> Tuple[List[DatasetPair], Dict[str, object]]:
    code_map = load_code_examples()

    candidate_pairs: List[DatasetPair] = []
    candidate_pairs.extend(load_canonical_pairs(code_map))
    candidate_pairs.extend(load_prompt_csv_pairs(DEFAULT_LOW_LEVEL_PROMPTS_CSV, "low_level_generated", code_map))

    if include_high_level:
        candidate_pairs.extend(
            load_prompt_csv_pairs(
                DEFAULT_HIGH_LEVEL_PROMPTS_CSV,
                "high_level_generated",
                code_map,
                require_specific_prompts=False,
            )
        )

    if include_generated_summaries and DEFAULT_SUMMARY_CSV.exists():
        with DEFAULT_SUMMARY_CSV.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                prompt = row.get("Question", "").strip()
                assistant = sanitize_assistant_code(row.get("Answer", ""))
                if prompt and assistant:
                    candidate_pairs.append(DatasetPair(prompt=prompt, assistant=assistant, source="generated_summary"))

    candidate_pairs.extend(load_verified_pairs())
    candidate_pairs.extend(load_generalized_verified_pairs())
    candidate_pairs.extend(load_curated_pairs())
    candidate_pairs.extend(load_few_shot_pairs())

    rejections: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    kept_pairs: List[DatasetPair] = []

    for pair in candidate_pairs:
        if pair.messages:
            pair.messages = normalize_messages(pair.messages)
            reason = validate_messages(pair.messages)
            if reason:
                rejections[reason] += 1
                continue
            kept_pairs.append(pair)
            by_source[pair.source] += 1
            continue

        pair.prompt = normalize_prompt(pair.prompt)
        pair.assistant = sanitize_assistant_code(pair.assistant)

        if not pair.prompt:
            rejections["empty_prompt"] += 1
            continue
        if not pair.assistant:
            rejections["empty_assistant"] += 1
            continue
        if any(marker in pair.prompt.lower() for marker in BANNED_PROMPT_MARKERS):
            rejections["banned_prompt_marker"] += 1
            continue

        reason = assistant_is_obviously_bad(pair.assistant)
        if reason:
            rejections[reason] += 1
            continue

        kept_pairs.append(pair)
        by_source[pair.source] += 1

    deduped_pairs, duplicate_count = dedupe_pairs(kept_pairs)
    rejections["exact_duplicates"] += duplicate_count

    audit = {
        "candidate_pairs": len(candidate_pairs),
        "accepted_pairs": len(deduped_pairs),
        "rejections": dict(rejections),
        "accepted_by_source": dict(by_source),
        "raw_jsonl_audit": audit_jsonl(DEFAULT_RAW_JSONL),
    }
    return deduped_pairs, audit


def write_jsonl(pairs: Iterable[DatasetPair], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            if pair.messages:
                payload = {"messages": pair.messages}
            else:
                payload = {
                    "messages": [
                        {"role": "user", "content": pair.prompt},
                        {"role": "assistant", "content": pair.assistant},
                    ]
                }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_audit(audit: Dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
