"""Structured ANDES state — dataclass + parsers/extractors that derive state
from user_context, plus report-kind inference and applicability gates.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. All
dependencies are already-extracted lower-level helpers
(extractors) plus the external ``src.agent_evolution`` profile overrides.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent_evolution import (
    get_profile_marker_overrides,
    get_profile_pattern_overrides,
)
from src.andes_code.extractors import (
    extract_effective_user_context,
    extract_uploaded_files_from_context,
    infer_requested_builtin_case,
)


@dataclass
class StructuredAndesState:
    case_source: str = ""
    case_reference: str = ""
    uploaded_filename: str = ""
    add_ops: List[Dict[str, Any]] = field(default_factory=list)
    scale_factor: Optional[float] = None
    slack_setpoint: Optional[float] = None
    pv_setpoint: Optional[float] = None
    bus_rank_count: Optional[int] = None
    line_rank_count: Optional[int] = None
    target_pq_bus: Optional[int] = None
    target_pq_scale_factor: Optional[float] = None
    target_pv_bus: Optional[int] = None
    target_pv_setpoint: Optional[float] = None
    opened_line_pair: Optional[Tuple[int, int]] = None
    n1_candidate_lines: List[Tuple[int, int]] = field(default_factory=list)


def extract_result_json_keys(user_context: str) -> List[str]:
    match = re.search(
        r"The JSON object must contain these keys:\s*([^\n]+)",
        user_context or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return []

    raw_keys = match.group(1).strip().rstrip(".")
    return [item.strip() for item in raw_keys.split(",") if item.strip()]


def extract_first_float(user_context: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, user_context or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_first_int(user_context: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, user_context or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_first_float_from_patterns(user_context: str, patterns: Tuple[str, ...]) -> Optional[float]:
    for pattern in patterns:
        value = extract_first_float(user_context, pattern)
        if value is not None:
            return value
    return None


def extract_first_int_from_patterns(user_context: str, patterns: Tuple[str, ...]) -> Optional[int]:
    for pattern in patterns:
        value = extract_first_int(user_context, pattern)
        if value is not None:
            return value
    return None


def extract_low_voltage_threshold(user_context: str) -> Optional[float]:
    return extract_first_float_from_patterns(
        user_context,
        (
            r"below ([0-9]*\.?[0-9]+) p\.u\.",
            r"under ([0-9]*\.?[0-9]+) p\.u\.",
            r"lower than ([0-9]*\.?[0-9]+) p\.u\.",
            r"less than ([0-9]*\.?[0-9]+) p\.u\.",
        ),
    )


def extract_high_voltage_threshold(user_context: str) -> Optional[float]:
    return extract_first_float_from_patterns(
        user_context,
        (
            r"above ([0-9]*\.?[0-9]+) p\.u\.",
            r"over ([0-9]*\.?[0-9]+) p\.u\.",
            r"higher than ([0-9]*\.?[0-9]+) p\.u\.",
            r"greater than ([0-9]*\.?[0-9]+) p\.u\.",
        ),
    )


def extract_top_k_from_prompt(user_context: str) -> Optional[int]:
    patterns = (
        r"top-(\d+)",
        r"top\s+(\d+)",
        r"report the (\d+) lowest-voltage buses",
        r"report the (\d+) highest-voltage buses",
    )
    for pattern in patterns:
        match = re.search(pattern, user_context or "", flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_plot_filename(user_context: str) -> str:
    match = re.search(r"to\s+'([^']+\.(?:png|jpg|jpeg|svg))'", user_context or "", flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'to\s+"([^"]+\.(?:png|jpg|jpeg|svg))"', user_context or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_add_pq_operation(user_context: str) -> Optional[Dict[str, Any]]:
    bus_match = re.search(
        r"add(?: one| a)?(?: new)? pq load(?: before setup)? at bus (\d+)",
        user_context or "",
        flags=re.IGNORECASE,
    )
    idx_match = re.search(r"idx ['\"]([^'\"]+)['\"]", user_context or "", flags=re.IGNORECASE)
    p0_match = re.search(r"\bp0\s*=\s*([0-9]*\.?[0-9]+)", user_context or "", flags=re.IGNORECASE)
    q0_match = re.search(r"\bq0\s*=\s*([0-9]*\.?[0-9]+)", user_context or "", flags=re.IGNORECASE)
    if not all([bus_match, idx_match, p0_match, q0_match]):
        return None

    return {
        "type": "add_pq",
        "bus": int(bus_match.group(1)),
        "idx": idx_match.group(1),
        "p0": float(p0_match.group(1)),
        "q0": float(q0_match.group(1)),
    }


def parse_target_pq_bus(user_context: str) -> Optional[int]:
    patterns = get_profile_pattern_overrides(
        "target_pq_bus",
        (
        r"pq load connected to bus (\d+)",
        r"pq load at bus (\d+)",
        r"existing pq load connected to bus (\d+)",
        r"existing pq load at bus (\d+)",
        r"demand element that is already tied to bus (\d+)",
        r"demand element tied to bus (\d+)",
        r"demand record belongs to bus (\d+)",
        r"record belongs to bus (\d+)",
        r"demand record sits on bus (\d+)",
        r"record sits on bus (\d+)",
        r"same demand record on bus (\d+)",
        r"demand record on bus (\d+)",
        r"load tied to bus (\d+)",
        r"load belongs to bus (\d+)",
        r"bus (\d+).*\bdemand record\b",
        ),
    )
    return extract_first_int_from_patterns(user_context, patterns)


def parse_target_pq_scale_factor(user_context: str) -> Optional[float]:
    factor = extract_first_float_from_patterns(
        user_context,
        tuple(
            get_profile_pattern_overrides(
                "target_pq_scale_factor",
                (
            r"scale both p0 and q0 of .*?load.*? by ([0-9]*\.?[0-9]+)",
            r"scale both p0 and q0 of that load by ([0-9]*\.?[0-9]+)",
            r"increase both active and reactive demand by a factor of ([0-9]*\.?[0-9]+)",
            r"increase both active and reactive demand by ([0-9]*\.?[0-9]+)",
            r"multiply both active and reactive demand by ([0-9]*\.?[0-9]+)",
            r"increase both p0 and q0 by (?:a factor of )?([0-9]*\.?[0-9]+)",
                ),
            )
        ),
    )
    if factor is not None:
        return factor

    percent = extract_first_float_from_patterns(
        user_context,
        tuple(
            get_profile_pattern_overrides(
                "target_pq_percent",
                (
            r"([0-9]*\.?[0-9]+)% heavier",
            r"([0-9]*\.?[0-9]+)% higher",
            r"([0-9]*\.?[0-9]+)% larger",
            r"([0-9]*\.?[0-9]+)% more on both active and reactive",
                ),
            )
        ),
    )
    if percent is None:
        return None
    return 1.0 + percent / 100.0


def parse_scale_pq_at_bus_operation(user_context: str) -> Optional[Dict[str, Any]]:
    target_bus = parse_target_pq_bus(user_context)
    factor = parse_target_pq_scale_factor(user_context)
    if target_bus is None or factor is None:
        return None
    return {"type": "scale_pq_at_bus", "bus": target_bus, "factor": factor}


def parse_target_pv_bus(user_context: str) -> Optional[int]:
    patterns = get_profile_pattern_overrides(
        "target_pv_bus",
        (
        r"pv device connected to bus (\d+)",
        r"pv device at bus (\d+)",
        r"generator voltage-control record tied to bus (\d+)",
        r"generator-side voltage-control record associated with bus (\d+)",
        r"voltage-control record associated with bus (\d+)",
        r"same voltage-control record on bus (\d+)",
        r"voltage-control record on bus (\d+)",
        r"voltage-controlled generator at bus (\d+)",
        r"generator regulating bus (\d+)",
        ),
    )
    return extract_first_int_from_patterns(user_context, patterns)


def parse_target_pv_v0_value(user_context: str) -> Optional[float]:
    return extract_first_float_from_patterns(
        user_context,
        tuple(
            get_profile_pattern_overrides(
                "target_pv_v0",
                (
            r"set its v0 target to ([0-9]*\.?[0-9]+)",
            r"move .* to a v0 target of ([0-9]*\.?[0-9]+)",
            r"move .* to a voltage target of ([0-9]*\.?[0-9]+)",
            r"move .* v0 target to ([0-9]*\.?[0-9]+)",
            r"move .* voltage target to ([0-9]*\.?[0-9]+)",
            r"adjust .* to a v0 target of ([0-9]*\.?[0-9]+)",
            r"adjust .* v0 target to ([0-9]*\.?[0-9]+)",
            r"raise .* to a v0 target of ([0-9]*\.?[0-9]+)",
            r"raise .* v0 target to ([0-9]*\.?[0-9]+)",
            r"raise .* regulator target to ([0-9]*\.?[0-9]+)",
            r"raise that regulator target to ([0-9]*\.?[0-9]+)",
            r"move that regulator target to ([0-9]*\.?[0-9]+)",
                ),
            )
        ),
    )


def parse_set_pv_bus_v0_operation(user_context: str) -> Optional[Dict[str, Any]]:
    target_bus = parse_target_pv_bus(user_context)
    value = parse_target_pv_v0_value(user_context)
    if target_bus is None or value is None:
        return None
    return {"type": "set_pv_bus_v0", "bus": target_bus, "value": value}


def parse_line_outage_by_pair(user_context: str) -> Optional[Tuple[int, int]]:
    patterns = get_profile_pattern_overrides(
        "line_outage_by_pair",
        (
        r"open the line between buses (\d+) and (\d+)",
        r"open the branch between buses (\d+) and (\d+)",
        r"take out the branch that links buses (\d+) and (\d+)",
        r"take out the branch linking buses (\d+) and (\d+)",
        r"trip the branch joining buses (\d+) and (\d+)",
        r"trip the branch between buses (\d+) and (\d+)",
        r"disconnect the branch between buses (\d+) and (\d+)",
        r"disconnect the branch linking buses (\d+) and (\d+)",
        r"remove the corridor between buses (\d+) and (\d+)",
        r"put the transmission corridor between buses (\d+) and (\d+) out of service",
        r"corridor between buses (\d+) and (\d+) out of service",
        r"knock the (\d+)-(\d+) corridor out of service",
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, user_context or "", flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def parse_candidate_line_pairs(user_context: str) -> List[Tuple[int, int]]:
    normalized = (user_context or "").lower()
    if not any(
        marker in normalized
        for marker in get_profile_marker_overrides(
            "candidate_line_markers",
            (
            "candidate lines",
            "candidate line",
            "candidate bus-pair list",
            "candidate bus pair list",
            "candidate branches",
            "branch pairs",
            "contingency list",
            "screening set",
            "outage set",
            ),
        )
    ):
        return []

    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for match in re.finditer(r"(\d+)\s*-\s*(\d+)", user_context or ""):
        pair = (int(match.group(1)), int(match.group(2)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def merge_structured_andes_state(
    current_state: Optional[StructuredAndesState],
    user_context: str,
) -> StructuredAndesState:
    effective_user_context = extract_effective_user_context(user_context)
    normalized_context = (effective_user_context or user_context or "").lower()
    state = StructuredAndesState() if current_state is None else StructuredAndesState(
        case_source=current_state.case_source,
        case_reference=current_state.case_reference,
        uploaded_filename=current_state.uploaded_filename,
        add_ops=[dict(item) for item in current_state.add_ops],
        scale_factor=current_state.scale_factor,
        slack_setpoint=current_state.slack_setpoint,
        pv_setpoint=current_state.pv_setpoint,
        bus_rank_count=current_state.bus_rank_count,
        line_rank_count=current_state.line_rank_count,
        target_pq_bus=current_state.target_pq_bus,
        target_pq_scale_factor=current_state.target_pq_scale_factor,
        target_pv_bus=current_state.target_pv_bus,
        target_pv_setpoint=current_state.target_pv_setpoint,
        opened_line_pair=current_state.opened_line_pair,
        n1_candidate_lines=list(current_state.n1_candidate_lines),
    )

    uploaded_files = extract_uploaded_files_from_context(user_context)
    if uploaded_files:
        state.case_source = "uploaded"
        state.uploaded_filename = os.path.basename(uploaded_files[0])
        state.case_reference = state.uploaded_filename
    else:
        builtin_case = infer_requested_builtin_case(effective_user_context or user_context)
        if builtin_case:
            state.case_source = "builtin"
            state.case_reference = builtin_case
            state.uploaded_filename = ""

    add_op = parse_add_pq_operation(effective_user_context or user_context)
    if add_op:
        state.add_ops = [item for item in state.add_ops if item.get("idx") != add_op["idx"]]
        state.add_ops.append(add_op)

    target_pq_bus = parse_target_pq_bus(effective_user_context or user_context)
    if target_pq_bus is not None:
        state.target_pq_bus = target_pq_bus

    scale_pq_op = parse_scale_pq_at_bus_operation(effective_user_context or user_context)
    if scale_pq_op:
        state.target_pq_bus = scale_pq_op["bus"]
        state.target_pq_scale_factor = scale_pq_op["factor"]
    else:
        inherited_pq_factor = parse_target_pq_scale_factor(effective_user_context or user_context)
        if inherited_pq_factor is not None and state.target_pq_bus is not None:
            if any(
                marker in normalized_context
                for marker in get_profile_marker_overrides(
                    "pq_carry_markers",
                    (
                        "same demand record",
                        "that same demand record",
                        "that demand record",
                        "same demand",
                        "demand components",
                    ),
                )
            ):
                state.target_pq_scale_factor = inherited_pq_factor

    scale_factor = extract_first_float(
        effective_user_context or user_context,
        r"scale (?:every|all) pq load(?:s)? by (?:a factor of )?([0-9]*\.?[0-9]+)",
    )
    if scale_factor is not None:
        state.scale_factor = scale_factor

    slack_setpoint = extract_first_float(
        effective_user_context or user_context,
        r"set the slack-bus voltage target to ([0-9]*\.?[0-9]+)",
    )
    if slack_setpoint is not None:
        state.slack_setpoint = slack_setpoint

    pv_setpoint = extract_first_float(
        effective_user_context or user_context,
        r"set the first pv voltage target to ([0-9]*\.?[0-9]+)",
    )
    if pv_setpoint is not None:
        state.pv_setpoint = pv_setpoint

    target_pv_bus = parse_target_pv_bus(effective_user_context or user_context)
    if target_pv_bus is not None:
        state.target_pv_bus = target_pv_bus

    targeted_pv_op = parse_set_pv_bus_v0_operation(effective_user_context or user_context)
    if targeted_pv_op:
        state.target_pv_bus = targeted_pv_op["bus"]
        state.target_pv_setpoint = targeted_pv_op["value"]
    else:
        inherited_pv_value = parse_target_pv_v0_value(effective_user_context or user_context)
        if inherited_pv_value is not None and state.target_pv_bus is not None:
            if any(
                marker in normalized_context
                for marker in get_profile_marker_overrides(
                    "pv_carry_markers",
                    (
                        "that regulator",
                        "same voltage-control record",
                        "that same voltage-control record",
                        "regulator target",
                        "that generator voltage-target change",
                        "keep that regulator change",
                    ),
                )
            ):
                state.target_pv_setpoint = inherited_pv_value

    opened_line_pair = parse_line_outage_by_pair(effective_user_context or user_context)
    if opened_line_pair:
        state.opened_line_pair = opened_line_pair

    n1_candidate_lines = parse_candidate_line_pairs(effective_user_context or user_context)
    if n1_candidate_lines:
        state.n1_candidate_lines = n1_candidate_lines

    top_k = extract_top_k_from_prompt(effective_user_context or user_context)
    if top_k is not None:
        normalized = (user_context or "").lower()
        result_keys = set(extract_result_json_keys(user_context))
        is_line_ranking_prompt = (
            "selected_line_ids" in result_keys
            or "selected_line_metrics" in result_keys
            or "phase angle" in normalized
            or bool(re.search(r"\btop[- ]\d+\s+lines?\b", normalized))
            or bool(re.search(r"report the \d+ lines?\b", normalized))
        )
        if is_line_ranking_prompt:
            state.line_rank_count = top_k
        else:
            state.bus_rank_count = top_k

    return state


def infer_structured_report_kind(result_keys: List[str], user_context: str) -> str:
    key_set = frozenset(result_keys)
    normalized = (user_context or "").lower()

    if key_set == frozenset({"slack_bus", "slack_voltage", "selected_bus_ids", "selected_voltages"}):
        return "baseline_high_rank_report"
    if key_set == frozenset({"target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"}):
        return "pq_bus_inspection_report"
    if key_set == frozenset({"target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"}):
        return "pq_bus_scale_report"
    if key_set == frozenset({"target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"}):
        return "pq_bus_scale_threshold_report"
    if key_set == frozenset({"scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_min_bus", "worst_min_voltage"}):
        return "n1_screening_report"
    if key_set == frozenset({"scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_outage_status", "worst_exit_code", "worst_island_count", "worst_no_slack_islands", "worst_islanded_bus_count", "worst_min_bus", "worst_min_voltage"}):
        return "n1_failure_aware_screening_report"
    if key_set == frozenset({"pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"}):
        return "pv_bus_inspection_report"
    if key_set == frozenset({"pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"}):
        return "pv_bus_adjust_threshold_report"
    if key_set == frozenset({"pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"}):
        return "pv_line_outage_report"
    if key_set == frozenset({"scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"}):
        return "pq_line_outage_threshold_report"
    if key_set == frozenset({"added_load_idx", "added_load_bus", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"}):
        return "add_load_threshold_report"
    if key_set == frozenset({"scale_factor", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"}):
        return "scaled_bar_plot_report" if "bar chart" in normalized or "bar plot" in normalized else "scaled_plot_report"
    if key_set == frozenset({"threshold", "selected_bus_ids", "selected_count", "lowest_bus_ids", "lowest_voltages"}):
        return "baseline_threshold_low_rank_report"
    if key_set == frozenset({"slack_bus", "slack_setpoint", "slack_voltage", "selected_count"}):
        return "slack_adjust_report"
    if key_set == frozenset({"added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "total_pq_count"}):
        return "extremes_report_with_total_pq"
    if key_set == frozenset({"pv_bus", "pv_setpoint", "pv_voltage", "selected_count"}):
        return "pv_adjust_report"
    if key_set == frozenset({"max_bus", "max_voltage", "min_bus", "min_voltage"}):
        return "extremes_report"
    if key_set == frozenset({"selected_bus_ids", "selected_voltages"}):
        return "baseline_low_rank_report"
    if key_set == frozenset({"added_load_idx", "slack_bus", "slack_voltage", "threshold", "selected_bus_ids", "selected_count"}):
        return "add_load_slack_threshold_report"
    if key_set == frozenset({"slack_setpoint", "slack_voltage", "selected_bus_ids", "selected_voltages", "plot_file"}):
        return "slack_plot_low_rank_report"
    if key_set == frozenset({"selected_line_ids", "selected_line_metrics"}):
        return "line_topk_report"
    if key_set == frozenset({"scale_factor", "angle_threshold", "selected_line_ids", "selected_count"}):
        return "scaled_line_threshold_report"
    if key_set == frozenset({"added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"}):
        return "add_load_voltage_plot_report"
    if key_set == frozenset({"slack_bus", "slack_voltage", "max_bus", "max_voltage", "min_bus", "min_voltage"}):
        return "baseline_slack_extremes_report"
    if key_set == frozenset({"slack_setpoint", "slack_voltage", "selected_line_ids", "selected_line_metrics"}):
        return "slack_line_topk_report"
    if key_set == frozenset({"slack_setpoint", "scale_factor", "angle_threshold", "selected_line_ids", "selected_count"}):
        return "slack_scaled_line_threshold_report"

    return ""


def structured_codegen_is_applicable(user_context: str) -> bool:
    normalized = (user_context or "").lower()
    if "result_json" not in normalized:
        return False
    has_solution_language = any(
        marker in normalized
        for marker in get_profile_marker_overrides(
            "structured_activation_markers",
            (
                "power flow",
                "power-flow",
                "pflow",
                "rerun",
                "after a power-flow solution",
                "candidate lines",
                "contingency list",
                "outage set",
                "stressed case",
            ),
        )
    ) or bool(re.search(r"\bsolve(?:d|s|ing)?\b", normalized))
    if not has_solution_language:
        return False
    return bool(extract_result_json_keys(user_context))


def structured_report_has_required_state(report_kind: str, state: StructuredAndesState) -> bool:
    if report_kind in {
        "pq_bus_inspection_report",
        "pq_bus_scale_report",
        "pq_bus_scale_threshold_report",
    }:
        if state.target_pq_bus is None:
            return False
        if report_kind in {"pq_bus_scale_report", "pq_bus_scale_threshold_report"}:
            return state.target_pq_scale_factor is not None
        return True

    if report_kind in {
        "pv_bus_inspection_report",
        "pv_bus_adjust_threshold_report",
    }:
        if state.target_pv_bus is None:
            return False
        if report_kind == "pv_bus_adjust_threshold_report":
            return state.target_pv_setpoint is not None
        return True

    if report_kind == "pv_line_outage_report":
        return (
            state.target_pv_bus is not None
            and state.target_pv_setpoint is not None
            and state.opened_line_pair is not None
        )

    if report_kind == "n1_screening_report":
        return (
            state.target_pq_bus is not None
            and state.target_pq_scale_factor is not None
            and bool(state.n1_candidate_lines)
        )

    if report_kind == "n1_failure_aware_screening_report":
        return (
            state.target_pq_bus is not None
            and state.target_pq_scale_factor is not None
            and bool(state.n1_candidate_lines)
        )

    if report_kind == "pq_line_outage_threshold_report":
        return (
            state.target_pq_bus is not None
            and state.target_pq_scale_factor is not None
            and state.opened_line_pair is not None
        )

    if report_kind in {
        "add_load_threshold_report",
        "extremes_report_with_total_pq",
        "add_load_slack_threshold_report",
        "add_load_voltage_plot_report",
    }:
        return bool(state.add_ops)

    if report_kind in {
        "scaled_plot_report",
        "scaled_bar_plot_report",
        "scaled_line_threshold_report",
    }:
        return state.scale_factor is not None

    if report_kind in {
        "slack_adjust_report",
        "slack_plot_low_rank_report",
        "slack_line_topk_report",
        "slack_scaled_line_threshold_report",
    }:
        return state.slack_setpoint is not None

    if report_kind == "pv_adjust_report":
        return state.pv_setpoint is not None

    return True
