"""Deterministic repairs applied to LLM-generated ANDES Python code.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. The core
is ``normalize_andes_code_block`` — a long sequence of regex-based
fix-ups keyed to specific known LLM mistakes (case-loading patterns,
unsupported attribute accessors, branch-flow API mapping, plot repair,
etc.). The orchestrator ``normalize_andes_response`` runs the block
transformer over every fenced ```python block in a response.

Dependencies (all already extracted in earlier Stage 1 batches):
  - src.andes_code.detectors: is_code_only_request, looks_like_python_script
  - src.andes_code.extractors: extract_python_code_blocks,
    extract_effective_user_context, extract_uploaded_files_from_context,
    infer_requested_builtin_case, extract_requested_bus_number
  - src.andes_case_catalog (external): get_andes_builtin_case_paths,
    suggest_andes_case_paths
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from src.andes_case_catalog import get_andes_builtin_case_paths, suggest_andes_case_paths
from src.andes_code.detectors import is_code_only_request, looks_like_python_script
from src.andes_code.extractors import (
    PYTHON_CODE_BLOCK_PATTERN,
    extract_effective_user_context,
    extract_python_code_blocks,
    extract_requested_bus_number,
    extract_uploaded_files_from_context,
    infer_requested_builtin_case,
)


def ensure_python_code_block(response_text: str, user_context: str = "") -> Tuple[str, List[str]]:
    """Repair missing or unfinished fenced Python code blocks for code-only requests."""
    normalized = response_text or ""
    notes: List[str] = []

    if "```python" in normalized and normalized.count("```") % 2 == 1:
        normalized = normalized.rstrip() + "\n```"
        notes.append("Closed an unfinished ```python fence.")

    if extract_python_code_blocks(normalized):
        return normalized, notes

    if is_code_only_request(user_context) and looks_like_python_script(normalized):
        normalized = f"```python\n{normalized.strip()}\n```"
        notes.append("Wrapped a plain Python response in a ```python``` code block.")

    return normalized, notes


def ensure_import(code: str, import_line: str) -> str:
    if import_line in code:
        return code

    lines = code.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("# required_dependencies:"):
        insert_at = 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1

    while insert_at < len(lines) and (
        lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")
    ):
        insert_at += 1

    lines.insert(insert_at, import_line)
    return "\n".join(lines)


def ensure_result_json_output(code: str, user_context: str = "") -> Tuple[str, List[str]]:
    normalized = code
    notes: List[str] = []
    if "result_json" not in (user_context or "").lower():
        return normalized, notes

    if "RESULT_JSON =" not in normalized and "result_json =" not in normalized:
        return normalized, notes

    normalized = ensure_import(normalized, "import json")
    result_var = "RESULT_JSON"
    if re.search(r"^\s*result_json\s*=", normalized, flags=re.MULTILINE) and not re.search(
        r"^\s*RESULT_JSON\s*=",
        normalized,
        flags=re.MULTILINE,
    ):
        result_var = "result_json"
    canonical_print = f'print("RESULT_JSON=" + json.dumps({result_var}, sort_keys=True))'

    replaced = False
    updated_lines: List[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("print(") and "RESULT_JSON" in stripped:
            if not replaced:
                updated_lines.append(canonical_print)
                replaced = True
            continue
        updated_lines.append(line)

    if not replaced:
        updated_lines.append(canonical_print)

    normalized = "\n".join(updated_lines)
    notes.append("Canonicalized RESULT_JSON printing with json.dumps for strict machine parsing.")
    return normalized, notes


def resolve_builtin_case_path(raw_path: str, builtin_case_paths: set[str]) -> str:
    normalized = (raw_path or "").replace("\\", "/")
    if normalized in builtin_case_paths:
        return normalized

    basename = os.path.basename(normalized)
    basename_matches = [path for path in builtin_case_paths if os.path.basename(path) == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]

    suggestions = suggest_andes_case_paths(normalized, max_suggestions=1)
    if suggestions:
        return suggestions[0]

    if basename != normalized:
        suggestions = suggest_andes_case_paths(basename, max_suggestions=1)
        if suggestions:
            return suggestions[0]

    return ""


def transform_python_code_blocks(text: str, transformer) -> Tuple[str, List[str]]:
    """Apply a transformer to every Python code block in a response."""
    notes: List[str] = []
    pattern = re.compile(PYTHON_CODE_BLOCK_PATTERN, re.DOTALL)

    if not pattern.search(text or ""):
        stripped = (text or "").strip()
        if looks_like_python_script(stripped):
            updated_code, code_notes = transformer(stripped)
            notes.extend(code_notes)
            return updated_code, notes
        return text, notes

    def _replace(match: re.Match) -> str:
        code = match.group(1).strip()
        updated_code, code_notes = transformer(code)
        notes.extend(code_notes)
        return f"```python\n{updated_code}\n```"

    return pattern.sub(_replace, text), notes


def normalize_andes_code_block(code: str, user_context: str = "") -> Tuple[str, List[str]]:
    """Repair a few high-confidence ANDES code-generation mistakes."""
    normalized = code
    notes: List[str] = []
    effective_user_context = extract_effective_user_context(user_context)
    uploaded_files = extract_uploaded_files_from_context(user_context)
    uploaded_file_set = {os.path.basename(name) for name in uploaded_files}
    builtin_case_paths = set(get_andes_builtin_case_paths())
    normalized_user_context = (effective_user_context or user_context or "").lower()
    builtin_case_request = ("built-in" in normalized_user_context or "builtin" in normalized_user_context) and not uploaded_file_set
    expected_builtin_case = infer_requested_builtin_case(effective_user_context or user_context)
    requested_bus_number = extract_requested_bus_number(effective_user_context or user_context)

    normalized, upper_pkg_count = re.subn(r"\bANDES\b", "andes", normalized)
    if upper_pkg_count:
        notes.append("Normalized `ANDES` imports/usages to lowercase `andes`.")

    normalized, pflow_count = re.subn(r"\.pflow\b", ".PFlow", normalized)
    if pflow_count:
        notes.append("Replaced lowercase `.pflow` with `.PFlow`.")

    normalized, np_os_count = re.subn(r"\bnp\.os\.", "os.", normalized)
    normalized, np_path_count = re.subn(r"\bnp\.path\.", "os.path.", normalized)
    if np_os_count or np_path_count:
        notes.append("Replaced invalid NumPy filesystem helpers with `os`/`os.path`.")

    normalized, block_comment_count = re.subn(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    normalized, slash_comment_count = re.subn(r"(^\s*)//", r"\1#", normalized, flags=re.MULTILINE)
    if block_comment_count or slash_comment_count:
        notes.append("Removed non-Python comment syntax from the generated script.")

    if "os.getcwd(" in normalized or "os.path.join(" in normalized:
        normalized = ensure_import(normalized, "import os")

    if "Bus.v.vn" in normalized:
        normalized = normalized.replace("Bus.v.vn", "Bus.v.v")
        notes.append("Replaced unsupported `Bus.v.vn` with `Bus.v.v`.")

    if "Bus.v.mag" in normalized:
        normalized = normalized.replace("Bus.v.mag", "Bus.v.v")
        notes.append("Replaced unsupported `Bus.v.mag` with `Bus.v.v`.")

    normalized, line_status_attr_count = re.subn(r"\.Line\.status\.v\b", ".Line.u.v", normalized)
    if line_status_attr_count:
        notes.append("Replaced unsupported `ssa.Line.status.v` with the in-service flag `ssa.Line.u.v`.")

    # set_status() is ANDES 2.0+; rewrite to the backward-compatible
    # set(src="u",...) form so the code runs on both 1.x and 2.0 envs.
    # Use scalar idx and value (not lists) to avoid a TypeError in
    # ANDES 2.0 where set(src="u") internally calls set_status().
    #   ssa.Line.set_status(line_id, 0)  ->  ssa.Line.set(src="u", idx=line_id, attr="v", value=0)
    normalized, set_status_count = re.subn(
        r"(?P<prefix>\.Line)\.set_status\(\s*(?P<id>[^,\)]+?)\s*,\s*(?P<val>[^\)]+?)\s*\)",
        r'\g<prefix>.set(src="u", idx=\g<id>, attr="v", value=\g<val>)',
        normalized,
    )
    if set_status_count:
        notes.append("Replaced `ssa.Line.set_status(id, v)` with the backward-compatible `ssa.Line.set(src=\"u\", idx=id, attr=\"v\", value=v)` form.")

    normalized, line_status_set_count = re.subn(
        r"\.Line\.set\(\s*src\s*=\s*(['\"])status\1",
        '.Line.set(src="u"',
        normalized,
    )
    if line_status_set_count:
        notes.append("Replaced unsupported line `status` setters with the runtime-supported in-service flag `src=\"u\"`.")

    normalized, line_set_false_list_count = re.subn(
        r"(\.Line\.set\([^\n]*src\s*=\s*[\"']u[\"'][^\n]*value\s*=\s*)\[\s*False\s*\]",
        r"\g<1>[0]",
        normalized,
    )
    normalized, line_set_false_scalar_count = re.subn(
        r"(\.Line\.set\([^\n]*src\s*=\s*[\"']u[\"'][^\n]*value\s*=\s*)False\b",
        r"\g<1>0",
        normalized,
    )
    if line_set_false_list_count or line_set_false_scalar_count:
        notes.append("Converted boolean line-outage values to numeric in-service flags expected by `ssa.Line.set(src=\"u\", ...)`.")

    # Unwrap list-wrapped idx and value in set(src="u", idx=[X], ..., value=[Y])
    # to scalar form idx=X, value=Y. ANDES 2.0 internally redirects
    # set(src="u") to set_status(), which crashes on list idx.
    # MUST run AFTER the status->u and False->0 fixers above so we don't
    # have to repeat their pattern conversion in this rule.
    normalized, unwrap_idx_count = re.subn(
        r'(\.Line\.set\([^\n]*src\s*=\s*["\']u["\'][^\n]*idx\s*=\s*)\[([^\]]+)\]',
        r"\1\2",
        normalized,
    )
    normalized, unwrap_val_count = re.subn(
        r'(\.Line\.set\([^\n]*src\s*=\s*["\']u["\'][^\n]*value\s*=\s*)\[([^\]]+)\]',
        r"\1\2",
        normalized,
    )
    if unwrap_idx_count or unwrap_val_count:
        notes.append("Unwrapped list-wrapped idx/value in `ssa.Line.set(src=\"u\", ...)` to scalar form for ANDES 2.0 compatibility.")

    normalized, stem_collection_count = re.subn(r",\s*use_line_collection\s*=\s*True", "", normalized)
    if stem_collection_count:
        notes.append("Removed unsupported `use_line_collection=True` from `plt.stem(...)` for local matplotlib compatibility.")

    normalized, bus_eval_count = re.subn(r"\.Bus\.v\.v\.e\b", ".Bus.v.v", normalized)
    normalized, line_sn_eval_count = re.subn(r"\.Line\.Sn\.e\b", ".Line.Sn.v", normalized)
    normalized, line_idx_eval_count = re.subn(r"\.Line\.idx\.e\b", ".Line.idx.v", normalized)
    normalized, bus_idx_name_count = re.subn(r"\.Bus\.idx\.v\.name\b", ".Bus.idx.v", normalized)
    if bus_eval_count or line_sn_eval_count or line_idx_eval_count or bus_idx_name_count:
        notes.append("Normalized unsupported `.e` / `.name` result accessors to ANDES 2.0 vector accessors.")

    branch_flow_request = any(token in normalized_user_context for token in ("line", "lines", "branch", "branches"))
    if branch_flow_request and "active power" in normalized_user_context:
        normalized, line_p1_count = re.subn(r"\.Line\.p1\.(?:v|e)\b", ".Line.a1.e", normalized)
        normalized, line_p2_count = re.subn(r"\.Line\.p2\.(?:v|e)\b", ".Line.a2.e", normalized)
        normalized, line_a1_v_count = re.subn(r"\.Line\.a1\.v\b", ".Line.a1.e", normalized)
        normalized, line_a2_v_count = re.subn(r"\.Line\.a2\.v\b", ".Line.a2.e", normalized)
        if line_p1_count or line_p2_count or line_a1_v_count or line_a2_v_count:
            notes.append("Mapped branch active-power flow requests to the supported ANDES arrays `ssa.Line.a1.e` / `ssa.Line.a2.e`.")

    if branch_flow_request and "reactive power" in normalized_user_context:
        normalized, line_q1_count = re.subn(r"\.Line\.q1\.(?:v|e)\b", ".Line.v1.e", normalized)
        normalized, line_q2_count = re.subn(r"\.Line\.q2\.(?:v|e)\b", ".Line.v2.e", normalized)
        normalized, line_v1_v_count = re.subn(r"\.Line\.v1\.v\b", ".Line.v1.e", normalized)
        normalized, line_v2_v_count = re.subn(r"\.Line\.v2\.v\b", ".Line.v2.e", normalized)
        if line_q1_count or line_q2_count or line_v1_v_count or line_v2_v_count:
            notes.append("Mapped branch reactive-power flow requests to the supported ANDES arrays `ssa.Line.v1.e` / `ssa.Line.v2.e`.")

    normalized, line_idx_numeric_cast_count = re.subn(
        r"np\.asarray\(\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*\.Line\.idx\.v)\s*,\s*dtype\s*=\s*(?:int|float|np\.[A-Za-z0-9_]+)\s*\)",
        r"[str(item) for item in np.asarray(\g<expr>)]",
        normalized,
    )
    if line_idx_numeric_cast_count:
        notes.append("Replaced numeric casts of `ssa.Line.idx.v` with string line-device IDs.")

    normalized, bus_v_count = re.subn(
        r"(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\.Bus\.v)(?!\.v)",
        r"\g<expr>.v",
        normalized,
    )
    if bus_v_count:
        notes.append("Expanded `ssa.Bus.v` to `ssa.Bus.v.v`.")

    normalized, bus_idx_count = re.subn(
        r"(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\.Bus\.idx)(?!\.v)",
        r"\g<expr>.v",
        normalized,
    )
    if bus_idx_count:
        notes.append("Expanded `ssa.Bus.idx` to `ssa.Bus.idx.v`.")

    normalized, slack_idx_count = re.subn(
        r"(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\.Slack\.bus)\.idx\.v",
        r"\g<expr>.v",
        normalized,
    )
    if slack_idx_count:
        notes.append("Replaced `ssa.Slack.bus.idx.v` with `ssa.Slack.bus.v`.")

    normalized, slack_idx_direct_count = re.subn(
        r"(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\.Slack\.bus)\.idx\b",
        r"\g<expr>.v",
        normalized,
    )
    if slack_idx_direct_count:
        notes.append("Replaced `ssa.Slack.bus.idx` with `ssa.Slack.bus.v`.")

    normalized, typo_bus_idx_count = re.subn(r"\bbuse_idx\b", "bus_idx", normalized)
    if typo_bus_idx_count:
        notes.append("Repaired the common `buse_idx` typo.")

    normalized, rc_run_count = re.subn(
        r"(^\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*\.PFlow\.run\(\))\s*$",
        r"\1\3",
        normalized,
        flags=re.MULTILINE,
    )
    normalized, rc_guard_count = re.subn(
        r"^\s*if\s+[A-Za-z_][A-Za-z0-9_]*\s*!=\s*0:\s*\n\s*raise RuntimeError\([^\n]*\)\s*\n?",
        "",
        normalized,
        flags=re.MULTILINE,
    )
    if rc_run_count or rc_guard_count:
        notes.append("Removed unsupported assumptions about `ssa.PFlow.run()` returning a convergence code.")

    normalized, add_count = re.subn(
        r"(\.add\()\s*model\s*=\s*([\"'][^\"']+[\"'])\s*,\s*",
        r"\1\2, ",
        normalized,
    )
    if add_count:
        notes.append("Rewrote `.add(model=...)` to `.add(<model_name>, ...)`.")

    def _replace_uploaded_get_case(match: re.Match) -> str:
        quote = match.group("quote")
        raw_path = match.group("path")
        basename = os.path.basename(raw_path)
        if basename in uploaded_file_set:
            notes.append(f"Replaced `andes.get_case(...)` with direct uploaded filename access for `{basename}`.")
            return f"{quote}{basename}{quote}"
        return match.group(0)

    normalized = re.sub(
        r"andes\.get_case\(\s*(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)\s*\)",
        _replace_uploaded_get_case,
        normalized,
    )

    def _repair_builtin_get_case(match: re.Match) -> str:
        quote = match.group("quote")
        raw_path = match.group("path").replace("\\", "/")
        basename = os.path.basename(raw_path)
        if basename in uploaded_file_set:
            return match.group(0)

        repaired_path = resolve_builtin_case_path(raw_path, builtin_case_paths)
        if repaired_path and repaired_path != raw_path:
            notes.append(f"Normalized built-in ANDES case path `{raw_path}` to `{repaired_path}`.")
            return f"andes.get_case({quote}{repaired_path}{quote})"
        return match.group(0)

    normalized = re.sub(
        r"andes\.get_case\(\s*(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)\s*\)",
        _repair_builtin_get_case,
        normalized,
    )

    if builtin_case_request and expected_builtin_case:
        normalized, expected_case_count = re.subn(
            r"andes\.get_case\(\s*[\"'][^\"']+[\"']\s*\)",
            f'andes.get_case("{expected_builtin_case}")',
            normalized,
            count=1,
        )
        if expected_case_count:
            notes.append(f"Aligned the built-in case path with the prompt-specific case `{expected_builtin_case}`.")

    def _replace_builtin_join(match: re.Match) -> str:
        quote = match.group("quote")
        raw_path = match.group("path").replace("\\", "/")
        basename = os.path.basename(raw_path)
        if not builtin_case_request or basename in uploaded_file_set:
            return match.group(0)

        repaired_path = resolve_builtin_case_path(raw_path, builtin_case_paths)
        if repaired_path:
            notes.append(f"Rewrote built-in case join path `{raw_path}` to `andes.get_case(\"{repaired_path}\")`.")
            return f"andes.get_case({quote}{repaired_path}{quote})"
        return match.group(0)

    normalized = re.sub(
        r"os\.path\.join\(\s*[^,\n]+,\s*(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)\s*\)",
        _replace_builtin_join,
        normalized,
    )

    def _replace_builtin_literal_assignment(match: re.Match) -> str:
        indent = match.group("indent")
        lhs = match.group("lhs")
        quote = match.group("quote")
        raw_path = match.group("path").replace("\\", "/")
        basename = os.path.basename(raw_path)
        if basename in uploaded_file_set:
            return match.group(0)

        repaired_path = expected_builtin_case if builtin_case_request and expected_builtin_case else resolve_builtin_case_path(raw_path, builtin_case_paths)
        if repaired_path:
            notes.append(f"Rewrote built-in case variable `{raw_path}` to use `andes.get_case(\"{repaired_path}\")`.")
            return f'{indent}{lhs} = andes.get_case({quote}{repaired_path}{quote})'
        return match.group(0)

    normalized = re.sub(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)\s*$",
        _replace_builtin_literal_assignment,
        normalized,
        flags=re.MULTILINE,
    )

    def _replace_builtin_literal_load(match: re.Match) -> str:
        quote = match.group("quote")
        raw_path = match.group("path").replace("\\", "/")
        basename = os.path.basename(raw_path)
        if basename in uploaded_file_set:
            return match.group(0)
        repaired_path = expected_builtin_case if builtin_case_request and expected_builtin_case else resolve_builtin_case_path(raw_path, builtin_case_paths)
        if repaired_path:
            if repaired_path == raw_path:
                notes.append(f"Wrapped built-in ANDES case `{raw_path}` with `andes.get_case(...)`.")
            else:
                notes.append(f"Normalized built-in ANDES case `{raw_path}` to `{repaired_path}` and wrapped it with `andes.get_case(...)`.")
            return f"andes.load(andes.get_case({quote}{repaired_path}{quote}),"
        return match.group(0)

    normalized = re.sub(
        r"andes\.load\(\s*(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)\s*,",
        _replace_builtin_literal_load,
        normalized,
    )

    def _wrap_simple_rhs_with_numpy(pattern: str, wrapper: str, note: str) -> None:
        nonlocal normalized
        updated, count = re.subn(pattern, wrapper, normalized, flags=re.MULTILINE)
        if count:
            normalized = updated
            notes.append(note)

    _wrap_simple_rhs_with_numpy(
        r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([A-Za-z_][A-Za-z0-9_]*\.Bus\.v\.v)(\s*(?:#.*)?)$",
        r"\1np.asarray(\2, dtype=float)\3",
        "Wrapped bus voltage arrays with `np.asarray(..., dtype=float)`.",
    )
    _wrap_simple_rhs_with_numpy(
        r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([A-Za-z_][A-Za-z0-9_]*\.Bus\.idx\.v)(\s*(?:#.*)?)$",
        r"\1np.asarray(\2, dtype=int)\3",
        "Wrapped bus index arrays with `np.asarray(..., dtype=int)`.",
    )
    _wrap_simple_rhs_with_numpy(
        r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([A-Za-z_][A-Za-z0-9_]*\.Line\.idx\.v)(\s*(?:#.*)?)$",
        r"\1[str(item) for item in np.asarray(\2)]\3",
        "Wrapped line index arrays with string device IDs from `ssa.Line.idx.v`.",
    )
    _wrap_simple_rhs_with_numpy(
        r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([A-Za-z_][A-Za-z0-9_]*\.Line\.(?:a1|a2|v1|v2)\.e)(\s*(?:#.*)?)$",
        r"\1np.asarray(\2, dtype=float)\3",
        "Wrapped line power arrays with `np.asarray(..., dtype=float)`.",
    )
    _wrap_simple_rhs_with_numpy(
        r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([A-Za-z_][A-Za-z0-9_]*\.Line\.Sn\.v)(\s*(?:#.*)?)$",
        r"\1np.asarray(\2, dtype=float)\3",
        "Wrapped line rating arrays with `np.asarray(..., dtype=float)`.",
    )

    if "np.asarray(" in normalized:
        normalized = ensure_import(normalized, "import numpy as np")

    if "before setup" in normalized_user_context and ".add(" in normalized:
        normalized, setup_false_count = re.subn(r"setup\s*=\s*True", "setup=False", normalized)
        if setup_false_count:
            notes.append("Changed `setup=True` to `setup=False` for add-before-setup workflows.")

        if ".setup()" not in normalized and ".PFlow.run()" in normalized:
            normalized = normalized.replace(".PFlow.run()", ".setup()\nssa.PFlow.run()", 1)
            notes.append("Inserted `.setup()` before `PFlow.run()` for an add-before-setup workflow.")

        if requested_bus_number and "\"bus\"" not in normalized and "bus=" not in normalized:
            normalized, bus_injected_count = re.subn(
                r'(\.add\(\s*[\"\']PQ[\"\']\s*,\s*param_dict\s*=\s*dict\()',
                rf'\1bus={requested_bus_number}, ',
                normalized,
                count=1,
            )
            if not bus_injected_count:
                normalized, bus_injected_count = re.subn(
                    r'(\.add\(\s*[\"\']PQ[\"\']\s*,\s*param_dict\s*=\s*\{)',
                    rf'\1"bus": {requested_bus_number}, ',
                    normalized,
                    count=1,
                )
            if bus_injected_count:
                notes.append(f"Injected the requested bus number `{requested_bus_number}` into the new PQ load definition.")

    if ".PFlow.plot_voltage()" in normalized:
        normalized = ensure_import(normalized, "import matplotlib.pyplot as plt")
        normalized = ensure_import(normalized, "import numpy as np")
        normalized, plot_fix_count = re.subn(
            r"(?P<ssa>[A-Za-z_][A-Za-z0-9_]*)\.PFlow\.plot_voltage\(\)",
            (
                "bus_id = np.asarray(\\g<ssa>.Bus.idx.v, dtype=float)\n"
                "bus_v = np.asarray(\\g<ssa>.Bus.v.v, dtype=float)\n"
                "plt.figure(figsize=(10, 4))\n"
                "plt.plot(bus_id, bus_v, marker=\"o\")\n"
                "plt.xlabel(\"Bus ID\")\n"
                "plt.ylabel(\"Voltage Magnitude (p.u.)\")\n"
                "plt.title(\"Voltage Profile\")\n"
                "plt.grid(True, alpha=0.3)\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            normalized,
        )
        if plot_fix_count:
            notes.append("Replaced unsupported `PFlow.plot_voltage()` with an explicit matplotlib voltage plot.")

    if ".TDS.plotter.plot(" in normalized or ".TDS.run()" in normalized:
        normalized = ensure_import(normalized, "import matplotlib.pyplot as plt")
        normalized = ensure_import(normalized, "import numpy as np")
        normalized, tds_run_count = re.subn(r"^\s*[A-Za-z_][A-Za-z0-9_]*\.TDS\.run\(\)\s*$\n?", "", normalized, flags=re.MULTILINE)
        normalized, tds_plot_count = re.subn(
            r"^\s*[A-Za-z_][A-Za-z0-9_]*\.TDS\.plotter\.plot\([^\n]+\)\s*$",
            (
                "bus_id = np.asarray(ssa.Bus.idx.v, dtype=float)\n"
                "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)\n"
                "plt.figure(figsize=(10, 4))\n"
                "plt.plot(bus_id, bus_v, marker=\"o\")\n"
                "plt.xlabel(\"Bus ID\")\n"
                "plt.ylabel(\"Voltage Magnitude (p.u.)\")\n"
                "plt.title(\"Voltage Profile\")\n"
                "plt.grid(True, alpha=0.3)\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            normalized,
            flags=re.MULTILINE,
        )
        if tds_run_count or tds_plot_count:
            notes.append("Replaced TDS plotting calls with a direct matplotlib power-flow voltage profile.")

    if "voltage profile" in normalized_user_context and "plt.bar(" in normalized and "plt.plot(" not in normalized:
        normalized = normalized.replace("plt.bar(", "plt.plot(", 1)
        notes.append("Converted a bar chart into a line-style voltage profile plot.")

    normalized, result_json_notes = ensure_result_json_output(normalized, user_context=user_context)
    notes.extend(result_json_notes)

    return normalized, notes


def normalize_andes_response(response_text: str, user_context: str = "") -> Tuple[str, List[str]]:
    """Apply deterministic ANDES repairs to every Python code block in a response."""
    normalized_response_text, notes = ensure_python_code_block(response_text, user_context=user_context)
    effective_user_context = extract_effective_user_context(user_context)
    transformed_text, transform_notes = transform_python_code_blocks(
        normalized_response_text,
        lambda code: normalize_andes_code_block(code, user_context=effective_user_context or user_context),
    )
    notes.extend(transform_notes)
    return transformed_text, notes
