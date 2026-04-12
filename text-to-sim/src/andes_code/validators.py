"""Validation passes over LLM-generated Python code for ANDES tasks.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. Three
passes:

- ``check_python_code_compilation`` — AST-level syntax check.
- ``validate_response_code``       — top-level check over every code
                                     block in a response.
- ``validate_andes_case_loading``  — the big pattern-based ruleset that
                                     rejects known ANDES API misuses.

Dependencies (all already extracted in earlier batches):
  - src.code_blocks (pre-existing): extract_python_code_segments
  - src.andes_code.detectors: prompt_explicitly_mentions_idx
  - src.andes_code.extractors: extract_effective_user_context,
    extract_uploaded_files_from_context, infer_requested_builtin_case,
    extract_requested_bus_number, extract_requested_bus_numbers
  - src.andes_case_catalog (external): get_andes_builtin_case_paths,
    suggest_andes_case_paths
"""

from __future__ import annotations

import ast
import os
import re
from typing import List, Tuple

from src.andes_case_catalog import get_andes_builtin_case_paths, suggest_andes_case_paths
from src.andes_code.detectors import prompt_explicitly_mentions_idx
from src.andes_code.extractors import (
    extract_effective_user_context,
    extract_requested_bus_number,
    extract_requested_bus_numbers,
    extract_uploaded_files_from_context,
    infer_requested_builtin_case,
)
from src.code_blocks import extract_python_code_segments


def check_python_code_compilation(code: str) -> Tuple[bool, str]:
    """
    Check if Python code compiles without syntax errors.
    Returns (is_valid, error_message)
    """
    try:
        # Parse the code to check for syntax errors
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax Error on line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\nProblematic line: {e.text.strip()}"
        return False, error_msg
    except Exception as e:
        return False, f"Compilation Error: {str(e)}"

def validate_response_code(response: str, user_context: str = "") -> Tuple[bool, List[str]]:
    """
    Validate all Python code blocks in a response.
    Returns (all_valid, error_messages)
    """
    code_segments = extract_python_code_segments(response)
    code_blocks = [segment.code for segment in code_segments]
    code_only_prompt = "runnable python code only" in user_context.lower() or "code only" in user_context.lower()
    effective_user_context = extract_effective_user_context(user_context)

    if not code_blocks:
        if "runnable python code only" in user_context.lower() or "code only" in user_context.lower():
            return False, ["Return one runnable Python script inside a ```python``` code block."]
        return True, []  # No code to validate

    if code_only_prompt and not any(segment.fenced for segment in code_segments):
        return False, ["Return one runnable Python script inside a ```python``` code block."]

    error_messages = []
    all_valid = True

    for i, code in enumerate(code_blocks):
        is_valid, error_msg = check_python_code_compilation(code)
        if not is_valid:
            all_valid = False
            error_messages.append(f"Code block {i+1}: {error_msg}")

        rule_errors = validate_andes_case_loading(
            code,
            user_context=effective_user_context or user_context,
        )
        if rule_errors:
            all_valid = False
            for rule_error in rule_errors:
                error_messages.append(f"Code block {i+1}: {rule_error}")

    return all_valid, error_messages


def validate_andes_case_loading(code: str, user_context: str = "") -> List[str]:
    """Validate common ANDES case-loading mistakes."""
    errors: List[str] = []
    effective_user_context = extract_effective_user_context(user_context)
    normalized_user_context = (effective_user_context or user_context or "").lower()
    uploaded_files = extract_uploaded_files_from_context(user_context)
    uploaded_file_set = {os.path.basename(name) for name in uploaded_files}
    expected_builtin_case = infer_requested_builtin_case(effective_user_context or user_context)
    requested_bus_number = extract_requested_bus_number(effective_user_context or user_context)

    if re.search(r"\bimport\s+anodes\b", code) or re.search(r"\banodes\.", code):
        errors.append("Use 'andes' package, not 'anodes'.")

    if re.search(r"\bimport\s+ANDES\b", code) or re.search(r"\bANDES\.", code):
        errors.append("Use lowercase `andes` imports and API calls, not `ANDES`.")

    get_case_args = re.findall(r'andes\.get_case\(\s*["\']([^"\']+)["\']\s*\)', code)
    invalid_uploaded_args = set()
    if uploaded_file_set and get_case_args:
        for arg in get_case_args:
            arg_basename = os.path.basename(arg)
            if arg_basename in uploaded_file_set:
                errors.append(
                    f"Uploaded case '{arg_basename}' must be loaded directly with andes.load(...), "
                    "not andes.get_case(...)."
                )
                invalid_uploaded_args.add(arg)
                break
            if "/" not in arg and "\\" not in arg and arg_basename.lower().endswith((".xlsx", ".xls", ".csv")):
                errors.append(
                    "When uploaded files are available, do not call andes.get_case('<filename>'). "
                    "Use andes.load('<exact_filename>', ...)."
                )
                invalid_uploaded_args.add(arg)
                break

    builtin_case_paths = set(get_andes_builtin_case_paths())
    literal_load_args = re.findall(r'andes\.load\(\s*["\']([^"\']+)["\']\s*,', code)
    for arg in literal_load_args:
        normalized_arg = arg.replace("\\", "/")
        basename = os.path.basename(normalized_arg)
        if basename in uploaded_file_set:
            continue
        if normalized_arg in builtin_case_paths:
            errors.append(
                f"Built-in case '{arg}' must be loaded with andes.load(andes.get_case('{arg}'), ...), "
                "not by passing a raw relative path directly into andes.load(...)."
            )

    if builtin_case_paths and get_case_args:
        for arg in get_case_args:
            if arg in invalid_uploaded_args:
                continue
            normalized_arg = arg.replace("\\", "/")
            if normalized_arg not in builtin_case_paths:
                suggestions = suggest_andes_case_paths(normalized_arg, max_suggestions=3)
                if suggestions:
                    errors.append(
                        f"'{arg}' is not a valid ANDES built-in case path for andes.get_case(...). "
                        f"Try one of: {', '.join(suggestions)}."
                    )
                else:
                    errors.append(
                        f"'{arg}' is not a valid ANDES built-in case path for andes.get_case(...). "
                        "Use an exact relative path under andes/cases."
                    )

    if "Bus.v.vn" in code:
        errors.append("Use `ssa.Bus.v.v` for bus voltage magnitudes in ANDES 2.0, not `Bus.v.vn`.")

    if "Bus.v.mag" in code:
        errors.append("Use `ssa.Bus.v.v` for bus voltage magnitudes in ANDES 2.0, not `Bus.v.mag`.")

    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.Bus\.v(?!\.v)", code):
        errors.append("Use `ssa.Bus.v.v`, not `ssa.Bus.v`, when you need bus voltage magnitudes.")

    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.Bus\.idx(?!\.v)", code):
        errors.append("Use `ssa.Bus.idx.v`, not `ssa.Bus.idx`, when you need bus IDs.")

    if re.search(r"\.add\(\s*model\s*=", code):
        errors.append(
            "Use `ssa.add(\"PQ\", param_dict=...)` or `ssa.add(model_name=\"PQ\", param_dict=...)`; "
            "`ssa.add(model=\"PQ\", ...)` is not valid in ANDES 2.0."
        )

    if re.search(r"\.add\(\s*[\"']PQ[\"']\s*,", code) and re.search(r"(^|[,{(]\s*)p\s*=", code):
        errors.append("For `PQ` devices, use `p0=` and `q0=` fields in `param_dict`, not `p=` or `q=`.")

    if re.search(r"for\s+\w+\s+in\s+\w+\.Bus\.idx\s*:", code):
        errors.append("Iterate over `ssa.Bus.idx.v`, not `ssa.Bus.idx`.")

    if ".Bus.slack" in code:
        errors.append("Use `ssa.Slack.bus.v[0]` to access the slack bus; `ssa.Bus.slack` is not a valid attribute.")

    if ".Slack.bus.idx" in code:
        errors.append("Use `ssa.Slack.bus.v[0]` to read the slack bus number, not `ssa.Slack.bus.idx`.")

    if ".pflow." in code:
        errors.append("Use `ssa.PFlow`, not `ssa.pflow`, for the power-flow routine.")

    if "plot_voltage(" in code:
        errors.append(
            "Do not call undocumented helpers like `PFlow.plot_voltage()`; build plots explicitly with matplotlib, "
            "`ssa.Bus.idx.v`, and `ssa.Bus.v.v`."
        )

    if "np._" in code:
        errors.append("Do not use private NumPy helper APIs such as `np._...`; use standard NumPy operations only.")

    if ("built-in" in normalized_user_context or "builtin" in normalized_user_context) and not uploaded_file_set:
        if "andes.get_case(" not in code:
            errors.append("Built-in case requests must load the case through `andes.get_case(...)`.")
        if expected_builtin_case and expected_builtin_case not in code:
            errors.append(
                f"The prompt asks for the built-in case `{expected_builtin_case}`. Load that exact ANDES case path."
            )

    if re.search(r"setup\s*=\s*False", code) and ".setup()" not in code:
        errors.append("When using `setup=False`, call `ssa.setup()` before running an ANDES routine.")

    if "before setup" in normalized_user_context:
        if "setup=True" in code:
            errors.append("For add-before-setup workflows, load the case with `setup=False` before calling `ssa.add(...)`.")
        if ".setup()" not in code:
            errors.append("For add-before-setup workflows, call `ssa.setup()` after `ssa.add(...)` and before `ssa.PFlow.run()`.")
        if requested_bus_number and ".add(" in code and "\"bus\"" not in code and "bus=" not in code:
            errors.append(f"The new PQ load should include the requested bus number `{requested_bus_number}` in the add call.")

    if ("runnable python code only" in normalized_user_context or "code only" in normalized_user_context):
        if re.search(r"\bimport\s+unittest\b|\bunittest\.main\(|\bTestCase\b|\bpytest\b", code):
            errors.append("Return one plain runnable Python script, not unittest/pytest scaffolding.")

    if "plot" in normalized_user_context:
        if not re.search(r"\bplt\.(plot|bar|scatter|step|stem|savefig|show)\s*\(|\.plot\s*\(", code):
            errors.append("The prompt asks for a plot, so the script should actually create a plot.")

    if "slack bus" in normalized_user_context and ".Slack." not in code:
        errors.append(
            "The prompt asks for slack bus results; use the ANDES Slack model (for example `ssa.Slack.bus.v`)."
        )
    elif "slack bus" in normalized_user_context and "ssa.Slack.bus.v[0]" not in code:
        errors.append("When the prompt asks for slack bus voltage, read the slack bus number with `ssa.Slack.bus.v[0]`.")

    if (
        ("line" in normalized_user_context or "lines" in normalized_user_context)
        and "angle" in normalized_user_context
    ):
        if "Line.a1.e" not in code:
            errors.append(
                "For line-angle analysis, use the ANDES line-angle result arrays such as `ssa.Line.a1.e`."
            )

    if re.search(r"np\.asarray\(\s*[^)]*\.Line\.idx\.v\s*,\s*dtype\s*=\s*(?:int|float|np\.[A-Za-z0-9_]+)\s*\)", code):
        errors.append(
            "`ssa.Line.idx.v` contains string device IDs such as `Line_1`; do not cast it to `int` or `float`. "
            "Use `[str(item) for item in np.asarray(ssa.Line.idx.v)]` instead."
        )

    if ".Line.status" in code:
        errors.append(
            "ANDES `Line` does not expose `status`; use `ssa.Line.u.v` if you need the in-service flag."
        )

    if re.search(r"\.Line\.set\(\s*src\s*=\s*[\"']status[\"']", code):
        errors.append(
            "Open a line in this runtime with `ssa.Line.set(src=\"u\", idx=[line_id], attr=\"v\", value=[0])`, not `src=\"status\"`."
        )

    if "use_line_collection=True" in code:
        errors.append(
            "Do not pass `use_line_collection=True` to `plt.stem()`; the local matplotlib version does not support it."
        )

    branch_context = any(token in normalized_user_context for token in ("line", "lines", "branch", "branches"))
    line_outage_prompt = branch_context and any(
        token in normalized_user_context
        for token in (
            "trip one line",
            "trip a line",
            "trip the line",
            "open one line",
            "open a line",
            "open the line",
            "disconnect the line",
            "line outage",
            "n-1",
            "n - 1",
            "contingency",
        )
    )
    contingency_screening_prompt = any(
        token in normalized_user_context
        for token in (
            "n-1",
            "n - 1",
            "contingency",
            "outage set",
            "screening set",
            "screen candidate lines",
        )
    )
    generic_branch_plot_prompt = (
        branch_context
        and "plot" in normalized_user_context
        and any(
            token in normalized_user_context
            for token in (
                "branch flow",
                "line flow",
                "branch active power",
                "active power flow of all the branches",
                "active power of all the branches",
            )
        )
        and not any(
            token in normalized_user_context
            for token in ("network diagram", "topology", "single-line", "one-line")
        )
    )
    if branch_context and "active power" in normalized_user_context:
        if re.search(r"\.Line\.p[12]\.(?:v|e)\b", code):
            errors.append(
                "ANDES `Line` does not expose `p1` / `p2`; use `ssa.Line.a1.e` or `ssa.Line.a2.e` for branch active-power flow."
            )
        if "Line.a1.e" not in code and "Line.a2.e" not in code:
            errors.append(
                "For branch active-power flow, use the supported ANDES arrays `ssa.Line.a1.e` or `ssa.Line.a2.e`."
            )

    if generic_branch_plot_prompt:
        if "Line.a1.e" not in code and "Line.a2.e" not in code:
            errors.append(
                "For a generic branch-flow plot, default to branch active-power flow using `ssa.Line.a1.e` / `ssa.Line.a2.e`."
            )
        if ".Line.bus1.v" in code or ".Line.bus2.v" in code:
            errors.append(
                "For a generic branch-flow plot, plot branch metrics against line IDs instead of drawing bus-to-bus segments. "
                "Use `line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]` with branch-flow arrays."
            )
        if re.search(r"\[\s*0\s*,\s*0\s*\]\s*\+\s*\d+\s*\*\s*\[", code):
            errors.append(
                "Do not build branch-flow plots with synthetic y-vectors like `[0, 0] + ...`; plot one value per branch."
            )

    if branch_context and "reactive power" in normalized_user_context:
        if re.search(r"\.Line\.q[12]\.(?:v|e)\b", code):
            errors.append(
                "ANDES `Line` does not expose `q1` / `q2`; use `ssa.Line.v1.e` or `ssa.Line.v2.e` for branch reactive-power flow."
            )
        if "Line.v1.e" not in code and "Line.v2.e" not in code:
            errors.append(
                "For branch reactive-power flow, use the supported ANDES arrays `ssa.Line.v1.e` or `ssa.Line.v2.e`."
            )

    if line_outage_prompt or contingency_screening_prompt:
        if re.search(r"\btrip_line\s*=\s*andes\.get_case\(", code):
            errors.append(
                "A tripped line must be selected from `ssa.Line.idx.v` or resolved from `ssa.Line.bus1.v` / `ssa.Line.bus2.v`; "
                "do not assign `andes.get_case(...)` to a line ID."
            )
        if line_outage_prompt and ".Line.idx.v" not in code:
            errors.append("Trip-line studies should inspect `ssa.Line.idx.v` to select a real line ID.")
        if line_outage_prompt and ".Line.set(" not in code and ".Line.set_status(" not in code:
            errors.append(
                "Trip-line studies should actually open a line with `ssa.Line.set_status(...)` or the legacy-equivalent "
                "`ssa.Line.set(src=\"u\", ...)` before rerunning power flow."
            )
        contingency_status_markers = 0
        if ".PFlow.converged" in code or re.search(r"\bconverged\s*=\s*bool\([^)]*PFlow\.run\(", code):
            contingency_status_markers += 1
        if ".exit_code" in code:
            contingency_status_markers += 1
        if ".Bus.island_sets" in code or re.search(r'getattr\(\s*[A-Za-z_][A-Za-z0-9_]*\.Bus\s*,\s*[\"\']island_sets[\"\']', code):
            contingency_status_markers += 1
        if ".Bus.nosw_island" in code or re.search(r'getattr\(\s*[A-Za-z_][A-Za-z0-9_]*\.Bus\s*,\s*[\"\']nosw_island[\"\']', code):
            contingency_status_markers += 1
        if ".Bus.n_islanded_buses" in code or re.search(r'getattr\(\s*[A-Za-z_][A-Za-z0-9_]*\.Bus\s*,\s*[\"\']n_islanded_buses[\"\']', code):
            contingency_status_markers += 1
        if contingency_status_markers < 2:
            errors.append(
                "For N-1 or line-outage studies, inspect post-contingency convergence and islanding with fields such as "
                "`ssa.PFlow.converged`, `ssa.exit_code`, `ssa.Bus.island_sets`, `ssa.Bus.nosw_island`, and "
                "`ssa.Bus.n_islanded_buses` before ranking the outage by voltage."
            )

    if "count" in normalized_user_context and not re.search(r"len\(|sum\(|np\.sum\(", code):
        errors.append("The prompt asks for a count, so compute and print an explicit count with `len(...)` or `sum(...)`.")

    if re.search(r"\btwo\b|\btop-2\b|\btop 2\b", normalized_user_context):
        if not re.search(r"argsort|sorted|argpartition", code):
            errors.append("The prompt asks for two ranked results; use sorting or argpartition to select them explicitly.")

    if re.search(r"\btop-3\b|\btop 3\b|\bthree\b", normalized_user_context):
        if not re.search(r"argsort|sorted|argpartition", code):
            errors.append("The prompt asks for three ranked results; use sorting or argpartition to select them explicitly.")

    modification_keywords = ("modify", "change", "update", "adjust", "set ")
    requested_bus_numbers = extract_requested_bus_numbers(user_context)
    explicit_idx_prompt = prompt_explicitly_mentions_idx(user_context)

    if requested_bus_numbers and any(keyword in normalized_user_context for keyword in modification_keywords):
        if ".PV.set(" in code and "first pv" not in normalized_user_context and not explicit_idx_prompt:
            if ".PV.bus.v" not in code or ".PV.idx.v" not in code:
                errors.append(
                    "When modifying an existing PV device by bus, inspect the case and resolve the real device idx "
                    "from `ssa.PV.bus.v` and `ssa.PV.idx.v` before calling `ssa.PV.set(...)`."
                )

        if (
            ".PQ.set(" in code
            and "scale every pq" not in normalized_user_context
            and "scale all pq" not in normalized_user_context
            and not explicit_idx_prompt
        ):
            if ".PQ.bus.v" not in code or ".PQ.idx.v" not in code:
                errors.append(
                    "When modifying an existing PQ load by bus, inspect the case and resolve the real device idx "
                    "from `ssa.PQ.bus.v` and `ssa.PQ.idx.v` before calling `ssa.PQ.set(...)`."
                )

        if ".Line.set(" in code and len(requested_bus_numbers) >= 2 and not explicit_idx_prompt:
            if ".Line.bus1.v" not in code or ".Line.bus2.v" not in code or ".Line.idx.v" not in code:
                errors.append(
                    "When modifying a line identified by terminal buses, inspect the case and resolve the real line idx "
                    "from `ssa.Line.bus1.v`, `ssa.Line.bus2.v`, and `ssa.Line.idx.v` before calling `ssa.Line.set(...)`."
                )

    return errors
