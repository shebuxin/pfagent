from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional


FULL_SUITE_SCENARIO_COUNT = 164
OPEN_GENERALIZATION_SCENARIO_COUNT = 5

CODE_ONLY_OPENERS = [
    "Return exactly one runnable Python code block and no prose.",
    "Write one complete runnable Python script only, inside a single ```python block.",
    "Give me code only: one full Python script in one fenced block.",
    "Please answer with one runnable Python script only and nothing else.",
]

FOLLOW_UP_OPENERS = [
    "Follow-up request: update the previous study and return a fresh complete script.",
    "Please revise the prior script for this next step and return one full script only.",
    "Next follow-up: rebuild the script for the updated study, code only.",
    "Please keep the conversation context and send one new complete script only.",
]


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _line_candidate(line_id: Any, bus1: int, bus2: int) -> Dict[str, Any]:
    return {"line_id": line_id, "bus1": int(bus1), "bus2": int(bus2)}


def _build_case_variants() -> List[Dict[str, Any]]:
    variants = [
        {
            "variant_id": "builtin_ieee14",
            "case_family": "ieee14",
            "case_source": "builtin",
            "source_case_path": "ieee14/ieee14_full.xlsx",
            "display_name": "the built-in IEEE 14 full case",
            "extension": ".xlsx",
            "add_buses": [4, 5, 9, 10],
            "threshold_low": [1.000, 1.005, 1.010, 1.015],
            "threshold_high": [1.015, 1.020, 1.025, 1.030],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [1.020, 1.025, 1.035, 1.040],
            "pv_values": [1.010, 1.015, 1.020, 1.025],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "uploaded_ieee14",
            "case_family": "ieee14",
            "case_source": "uploaded",
            "source_case_path": "ieee14/ieee14_full.xlsx",
            "display_name": "the uploaded IEEE 14 study file",
            "extension": ".xlsx",
            "add_buses": [4, 5, 9, 10],
            "threshold_low": [1.000, 1.005, 1.010, 1.015],
            "threshold_high": [1.015, 1.020, 1.025, 1.030],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [1.020, 1.025, 1.035, 1.040],
            "pv_values": [1.010, 1.015, 1.020, 1.025],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "builtin_ieee39",
            "case_family": "ieee39",
            "case_source": "builtin",
            "source_case_path": "ieee39/ieee39.xlsx",
            "display_name": "the built-in IEEE 39 case",
            "extension": ".xlsx",
            "add_buses": [4, 10, 15, 20],
            "threshold_low": [0.950, 0.960, 0.970, 0.980],
            "threshold_high": [1.020, 1.030, 1.040, 1.050],
            "scale_values": [1.020, 1.030, 1.040, 1.050],
            "slack_values": [1.015, 1.020, 1.030, 1.035],
            "pv_values": [1.005, 1.010, 1.015, 1.020],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "uploaded_ieee39",
            "case_family": "ieee39",
            "case_source": "uploaded",
            "source_case_path": "ieee39/ieee39.xlsx",
            "display_name": "the uploaded IEEE 39 study file",
            "extension": ".xlsx",
            "add_buses": [4, 10, 15, 20],
            "threshold_low": [0.950, 0.960, 0.970, 0.980],
            "threshold_high": [1.020, 1.030, 1.040, 1.050],
            "scale_values": [1.020, 1.030, 1.040, 1.050],
            "slack_values": [1.015, 1.020, 1.030, 1.035],
            "pv_values": [1.005, 1.010, 1.015, 1.020],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "builtin_kundur",
            "case_family": "kundur",
            "case_source": "builtin",
            "source_case_path": "kundur/kundur_full.xlsx",
            "display_name": "the built-in Kundur full case",
            "extension": ".xlsx",
            "add_buses": [4, 6, 7, 9],
            "threshold_low": [0.940, 0.950, 0.960, 0.970],
            "threshold_high": [0.990, 1.000, 1.010, 1.020],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [0.990, 1.000, 1.010, 1.020],
            "pv_values": [0.990, 1.000, 1.010, 1.020],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "uploaded_kundur",
            "case_family": "kundur",
            "case_source": "uploaded",
            "source_case_path": "kundur/kundur_full.xlsx",
            "display_name": "the uploaded Kundur study file",
            "extension": ".xlsx",
            "add_buses": [4, 6, 7, 9],
            "threshold_low": [0.940, 0.950, 0.960, 0.970],
            "threshold_high": [0.990, 1.000, 1.010, 1.020],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [0.990, 1.000, 1.010, 1.020],
            "pv_values": [0.990, 1.000, 1.010, 1.020],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "builtin_pjm5",
            "case_family": "pjm5",
            "case_source": "builtin",
            "source_case_path": "5bus/pjm5bus.json",
            "display_name": "the built-in PJM 5-bus case",
            "extension": ".json",
            "add_buses": [1, 2, 3, 4],
            "voltage_threshold": [0.980, 0.990, 1.000, 1.010],
            "angle_threshold": [0.080, 0.100, 0.120, 0.150],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [1.000, 1.010, 1.020, 1.030],
            "top_k": [2, 3, 4, 5],
        },
        {
            "variant_id": "uploaded_pjm5",
            "case_family": "pjm5",
            "case_source": "uploaded",
            "source_case_path": "5bus/pjm5bus.json",
            "display_name": "the uploaded PJM 5-bus study file",
            "extension": ".json",
            "add_buses": [1, 2, 3, 4],
            "voltage_threshold": [0.980, 0.990, 1.000, 1.010],
            "angle_threshold": [0.080, 0.100, 0.120, 0.150],
            "scale_values": [1.030, 1.040, 1.050, 1.060],
            "slack_values": [1.000, 1.010, 1.020, 1.030],
            "top_k": [2, 3, 4, 5],
        },
    ]

    extra_by_family = {
        "ieee14": {
            "pq_target_buses": [2, 3, 4, 5],
            "pv_target_buses": [2, 3, 6, 8],
            "line_candidates": [
                _line_candidate("Line_1", 1, 2),
                _line_candidate("Line_2", 1, 5),
                _line_candidate("Line_3", 2, 3),
                _line_candidate("Line_4", 2, 4),
            ],
            "stress_line_candidates": [
                _line_candidate("Line_10", 6, 13),
                _line_candidate("Line_12", 9, 10),
                _line_candidate("Line_20", 8, 7),
                _line_candidate("Line_9", 6, 12),
            ],
        },
        "ieee39": {
            "pq_target_buses": [3, 4, 7, 8],
            "pv_target_buses": [30, 31, 32, 33],
            "line_candidates": [
                _line_candidate("Line_1", 1, 2),
                _line_candidate("Line_2", 1, 39),
                _line_candidate("Line_3", 2, 3),
                _line_candidate("Line_4", 2, 25),
            ],
            "stress_line_candidates": [
                _line_candidate("Line_22", 16, 19),
                _line_candidate("Line_11", 6, 7),
                _line_candidate("Line_46", 29, 38),
                _line_candidate("Line_24", 16, 24),
            ],
        },
        "kundur": {
            "pq_target_buses": [7, 8, 7, 8],
            "pv_target_buses": [2, 3, 4, 2],
            "line_candidates": [
                _line_candidate("Line_0", 5, 6),
                _line_candidate("Line_2", 6, 7),
                _line_candidate("Line_4", 7, 8),
                _line_candidate("Line_7", 8, 9),
            ],
            "stress_line_candidates": [
                _line_candidate("Line_11", 1, 5),
                _line_candidate("Line_14", 4, 10),
                _line_candidate("Line_12", 2, 6),
                _line_candidate("Line_13", 3, 9),
            ],
        },
        "pjm5": {
            "pq_target_buses": [1, 2, 3, 1],
            "pv_target_buses": [0, 2, 4, 0],
            "pv_values": [1.010, 1.015, 1.020, 1.025],
            "line_candidates": [
                _line_candidate(0, 0, 1),
                _line_candidate(1, 0, 3),
                _line_candidate(2, 0, 4),
                _line_candidate(3, 1, 2),
            ],
            "stress_line_candidates": [
                _line_candidate(3, 1, 2),
                _line_candidate(1, 0, 3),
                _line_candidate(2, 0, 4),
                _line_candidate(0, 0, 1),
            ],
        },
    }

    for variant in variants:
        variant.update(copy.deepcopy(extra_by_family[variant["case_family"]]))

    return variants


def _uploaded_filename(variant: Dict[str, Any], scenario_number: int) -> Optional[str]:
    if variant["case_source"] != "uploaded":
        return None
    return f"verify_{variant['case_family']}_{scenario_number:03d}{variant['extension']}"


def _case_phrase(variant: Dict[str, Any], uploaded_filename: Optional[str], explicit: bool) -> str:
    if variant["case_source"] == "builtin":
        return (
            f"Use {variant['display_name']}."
            if explicit
            else "Keep using the same built-in case from earlier in this conversation."
        )
    if explicit:
        return f"Use my uploaded file {uploaded_filename} from the current working directory."
    return "Keep using the same uploaded study file from earlier in this conversation."


def _contract_lines(keys: List[str], extra: List[str]) -> str:
    key_list = ", ".join(keys)
    lines = [
        "The script must end by printing exactly one line that starts with RESULT_JSON=",
        f"The JSON object must contain these keys: {key_list}.",
        "Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.",
        "Round float values in RESULT_JSON to 6 decimals.",
    ]
    lines.extend(extra)
    return "\n".join(f"- {line}" for line in lines)


def _high_threshold_values(variant: Dict[str, Any]) -> List[float]:
    return variant.get("threshold_high", variant.get("voltage_threshold", []))


def _low_threshold_values(variant: Dict[str, Any]) -> List[float]:
    return variant.get("threshold_low", variant.get("voltage_threshold", []))


def _line_pair_text(line: Dict[str, Any]) -> str:
    return f"{line['bus1']}-{line['bus2']}"


def _line_pair_list_text(lines: List[Dict[str, Any]]) -> str:
    return ", ".join(_line_pair_text(line) for line in lines)


def _make_turn(
    turn_number: int,
    prompt: str,
    report_kind: str,
    result_keys: List[str],
    current_ops: List[Dict[str, Any]],
    cumulative_ops: List[Dict[str, Any]],
    current_checks: List[Dict[str, Any]],
    carry_checks: List[Dict[str, Any]],
    forbidden_patterns: List[str],
    plot_filename: Optional[str] = None,
    task_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "turn_id": turn_number,
        "prompt": prompt,
        "report_kind": report_kind,
        "result_keys": result_keys,
        "current_operations": copy.deepcopy(current_ops),
        "cumulative_operations": copy.deepcopy(cumulative_ops),
        "current_code_checks": copy.deepcopy(current_checks),
        "carry_forward_checks": copy.deepcopy(carry_checks),
        "forbidden_patterns": forbidden_patterns[:],
        "plot_filename": plot_filename,
        "expects_plot": plot_filename is not None,
        "task_params": dict(task_params or {}),
    }


def _build_common_checks(
    variant: Dict[str, Any],
    uploaded_filename: Optional[str],
    current_ops: List[Dict[str, Any]],
    carry_ops: List[Dict[str, Any]],
    plot_filename: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    current_checks: List[Dict[str, Any]] = []
    carry_checks: List[Dict[str, Any]] = []

    case_literal = uploaded_filename if variant["case_source"] == "uploaded" else variant["source_case_path"]
    current_checks.append(
        {"label": "case reference", "pattern": re.escape(case_literal), "weight": 2.0}
    )
    if variant["case_source"] == "builtin":
        current_checks.append(
            {"label": "built-in case loading", "pattern": r"andes\.get_case\(", "weight": 1.0}
        )
    else:
        current_checks.append(
            {"label": "uploaded case loading", "pattern": r"andes\.load\(", "weight": 1.0}
        )

    if plot_filename:
        current_checks.append(
            {"label": "plot save filename", "pattern": re.escape(plot_filename), "weight": 1.0}
        )
        current_checks.append(
            {"label": "plot save call", "pattern": r"savefig\(", "weight": 1.0}
        )

    def op_checks(op: Dict[str, Any]) -> List[Dict[str, Any]]:
        if op["type"] == "add_pq":
            return [
                {"label": "add PQ call", "pattern": r"\.add\(\s*[\"']PQ[\"']", "weight": 1.5},
                {"label": "added load idx", "pattern": re.escape(op["idx"]), "weight": 1.0},
                {"label": "added load bus", "pattern": rf"\bbus\b[^\n]*\b{op['bus']}\b", "weight": 1.0},
                {"label": "added load p0", "pattern": re.escape(_fmt(op["p0"])), "weight": 1.0},
                {"label": "added load q0", "pattern": re.escape(_fmt(op["q0"])), "weight": 1.0},
            ]
        if op["type"] == "scale_all_pq":
            return [
                {"label": "PQ scale call", "pattern": r"PQ\.set\(", "weight": 1.5},
                {"label": "PQ scale factor", "pattern": re.escape(_fmt(op["factor"])), "weight": 1.0},
            ]
        if op["type"] == "set_slack_v0":
            return [
                {"label": "Slack set call", "pattern": r"Slack\.set\(", "weight": 1.5},
                {"label": "Slack setpoint", "pattern": re.escape(_fmt(op["value"])), "weight": 1.0},
            ]
        if op["type"] == "set_first_pv_v0":
            return [
                {"label": "PV set call", "pattern": r"PV\.set\(", "weight": 1.5},
                {"label": "PV setpoint", "pattern": re.escape(_fmt(op["value"])), "weight": 1.0},
            ]
        if op["type"] == "scale_pq_at_bus":
            return [
                {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
                {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
                {"label": "PQ targeted set call", "pattern": r"PQ\.set\(", "weight": 1.5},
                {"label": "target PQ bus", "pattern": rf"\b{op['bus']}\b", "weight": 1.0},
                {"label": "target PQ factor", "pattern": re.escape(_fmt(op["factor"])), "weight": 1.0},
            ]
        if op["type"] == "set_pv_bus_v0":
            return [
                {"label": "PV bus array", "pattern": r"PV\.bus\.v", "weight": 1.0},
                {"label": "PV idx array", "pattern": r"PV\.idx\.v", "weight": 1.0},
                {"label": "PV targeted set call", "pattern": r"PV\.set\(", "weight": 1.5},
                {"label": "target PV bus", "pattern": rf"\b{op['bus']}\b", "weight": 1.0},
                {"label": "target PV setpoint", "pattern": re.escape(_fmt(op["value"])), "weight": 1.0},
            ]
        if op["type"] == "line_outage_by_pair":
            return [
                {"label": "line bus1 array", "pattern": r"Line\.bus1\.v", "weight": 1.0},
                {"label": "line bus2 array", "pattern": r"Line\.bus2\.v", "weight": 1.0},
                {"label": "line outage set call", "pattern": r"Line\.(?:set|set_status)\(", "weight": 1.5},
                {"label": "line outage in-service flag", "pattern": r"(?:src\s*=\s*[\"']u[\"']|Line\.set_status\()", "weight": 1.0},
                {"label": "outage bus1", "pattern": rf"\b{op['bus1']}\b", "weight": 1.0},
                {"label": "outage bus2", "pattern": rf"\b{op['bus2']}\b", "weight": 1.0},
            ]
        if op["type"] == "n1_screening":
            checks = [
                {"label": "line bus1 array", "pattern": r"Line\.bus1\.v", "weight": 1.0},
                {"label": "line bus2 array", "pattern": r"Line\.bus2\.v", "weight": 1.0},
                {"label": "line outage set call", "pattern": r"Line\.(?:set|set_status)\(", "weight": 1.5},
                {"label": "line outage in-service flag", "pattern": r"(?:src\s*=\s*[\"']u[\"']|Line\.set_status\()", "weight": 1.0},
            ]
            for candidate in op["candidate_lines"]:
                checks.extend(
                    [
                        {"label": f"candidate bus1 {_line_pair_text(candidate)}", "pattern": rf"\b{candidate['bus1']}\b", "weight": 0.5},
                        {"label": f"candidate bus2 {_line_pair_text(candidate)}", "pattern": rf"\b{candidate['bus2']}\b", "weight": 0.5},
                    ]
                )
            return checks
        return []

    for op in current_ops:
        current_checks.extend(op_checks(op))
    for op in carry_ops:
        carry_checks.extend(op_checks(op))

    forbidden_patterns = [
        r"\bunittest\b",
        r"\bpytest\b",
        r"\bimport\s+ANDES\b",
        r"ssa\.add\(\s*model\s*=",
    ]
    if variant["case_source"] == "uploaded" and uploaded_filename:
        forbidden_patterns.extend(
            [
                rf"andes\.get_case\(\s*[\"']{uploaded_filename}[\"']\s*\)",
                r"andes\.get_case\(\s*[\"'][^\"']+\.(?:xlsx|xls|json)[\"']\s*\)",
            ]
        )

    return {
        "current_checks": current_checks,
        "carry_checks": carry_checks,
        "forbidden_patterns": forbidden_patterns,
    }


def _build_voltage_blueprint_a(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    top_k = variant["top_k"][local_index]
    add_bus = variant["add_buses"][local_index]
    threshold = variant["threshold_low"][local_index]
    scale = variant["scale_values"][local_index]
    p0 = round(0.010 + 0.003 * local_index, 3)
    q0 = round(0.006 + 0.002 * local_index, 3)
    add_idx = f"PQ_VERIFY_{scenario_number:03d}_A"
    plot_file = f"scenario_{scenario_number:03d}_turn3_line.png"

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[scenario_number % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow and report the top-{top_k} highest-voltage buses plus the slack-bus voltage.",
            _contract_lines(
                ["slack_bus", "slack_voltage", "selected_bus_ids", "selected_voltages"],
                ["`selected_bus_ids` and `selected_voltages` should represent the top highest-voltage buses in descending order."],
            ),
        ]
    )

    current_ops_t2 = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same case, add one new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
            f"After rerunning power flow, report every bus below {threshold:.3f} p.u. together with the minimum-voltage bus.",
            _contract_lines(
                ["added_load_idx", "added_load_bus", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order."],
            ),
        ]
    )

    current_ops_t3 = [{"type": "scale_all_pq", "factor": scale}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep the added load from the previous step.",
            f"Also scale every PQ load by a factor of {scale:.3f} after setup, rerun power flow, and save a line plot of bus voltage magnitude to '{plot_file}'.",
            _contract_lines(
                ["scale_factor", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
                ["`plot_file` must exactly match the saved filename."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, plot_file)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "baseline_high_rank_report",
            ["slack_bus", "slack_voltage", "selected_bus_ids", "selected_voltages"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"top_k": top_k},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "add_load_threshold_report",
            ["added_load_idx", "added_load_bus", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "scaled_plot_report",
            ["scale_factor", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            plot_filename=plot_file,
            task_params={"plot_style": "line"},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "voltage_rank_add_scale_plot",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_voltage_blueprint_b(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    threshold_high = variant["threshold_high"][local_index]
    threshold_low = variant["threshold_low"][local_index]
    slack_value = variant["slack_values"][local_index]
    add_bus = variant["add_buses"][local_index]
    p0 = round(0.012 + 0.002 * local_index, 3)
    q0 = round(0.007 + 0.002 * local_index, 3)
    add_idx = f"PQ_VERIFY_{scenario_number:03d}_B"

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 1) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow, count all buses above {threshold_high:.3f} p.u., and also return the two lowest-voltage buses.",
            _contract_lines(
                ["threshold", "selected_bus_ids", "selected_count", "lowest_bus_ids", "lowest_voltages"],
                ["`selected_bus_ids` should list every bus above the threshold.", "`lowest_bus_ids` should contain exactly two buses in ascending voltage order."],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_slack_v0", "value": slack_value}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and report the slack bus voltage plus how many buses fall below {threshold_low:.3f} p.u.",
            _contract_lines(
                ["slack_bus", "slack_setpoint", "slack_voltage", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
            "Keep the adjusted slack-bus setting from the last turn.",
            f"Also add a new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
            "Rerun power flow and report the maximum-voltage bus, minimum-voltage bus, and the total number of PQ loads now present.",
            _contract_lines(
                ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "total_pq_count"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "baseline_threshold_low_rank_report",
            ["threshold", "selected_bus_ids", "selected_count", "lowest_bus_ids", "lowest_voltages"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"threshold": threshold_high},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "slack_adjust_report",
            ["slack_bus", "slack_setpoint", "slack_voltage", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"threshold": threshold_low},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "extremes_report",
            ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "total_pq_count"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "threshold_slack_add_extremes",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_voltage_blueprint_c(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    pv_value = variant["pv_values"][local_index]
    threshold = variant["threshold_high"][local_index]
    scale = variant["scale_values"][local_index]
    plot_file = f"scenario_{scenario_number:03d}_turn3_bar.png"

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 2) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            "Run power flow and report the maximum-voltage bus and minimum-voltage bus.",
            _contract_lines(
                ["max_bus", "max_voltage", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_first_pv_v0", "value": pv_value}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[scenario_number % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same case, set the first PV voltage target to {pv_value:.3f}, rerun power flow, and report the affected PV bus voltage together with how many buses are above {threshold:.3f} p.u.",
            _contract_lines(
                ["pv_bus", "pv_setpoint", "pv_voltage", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "scale_all_pq", "factor": scale}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            "Keep the PV setpoint adjustment from the previous turn.",
            f"Also scale every PQ load by {scale:.3f}, rerun power flow, and save a bar chart of the bus voltages to '{plot_file}'.",
            _contract_lines(
                ["scale_factor", "min_bus", "min_voltage", "max_bus", "max_voltage", "plot_file"],
                ["Use a bar chart, not a line chart."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, plot_file)
    checks_t3["current_checks"].append(
        {"label": "bar plot call", "pattern": r"plt\.bar\(", "weight": 1.0}
    )

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "extremes_report",
            ["max_bus", "max_voltage", "min_bus", "min_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pv_adjust_report",
            ["pv_bus", "pv_setpoint", "pv_voltage", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "scaled_bar_plot_report",
            ["scale_factor", "min_bus", "min_voltage", "max_bus", "max_voltage", "plot_file"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            plot_filename=plot_file,
            task_params={"plot_style": "bar"},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "extremes_pv_scale_barplot",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_voltage_blueprint_d(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    top_k = variant["top_k"][local_index]
    add_bus = variant["add_buses"][local_index]
    threshold = variant["threshold_low"][local_index]
    slack_value = variant["slack_values"][local_index]
    p0 = round(0.014 + 0.002 * local_index, 3)
    q0 = round(0.008 + 0.002 * local_index, 3)
    add_idx = f"PQ_VERIFY_{scenario_number:03d}_D"
    plot_file = f"scenario_{scenario_number:03d}_turn3_line.png"

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 3) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow and report the {top_k} lowest-voltage buses.",
            _contract_lines(
                ["selected_bus_ids", "selected_voltages"],
                ["`selected_bus_ids` and `selected_voltages` should represent the lowest-voltage buses in ascending voltage order."],
            ),
        ]
    )

    current_ops_t2 = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study and add a new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
            f"After rerunning, report the slack-bus voltage and every bus below {threshold:.3f} p.u.",
            _contract_lines(
                ["added_load_idx", "slack_bus", "slack_voltage", "threshold", "selected_bus_ids", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "set_slack_v0", "value": slack_value}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep the added load from the previous turn.",
            f"Also set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and save a line plot of bus voltages to '{plot_file}'.",
            _contract_lines(
                ["slack_setpoint", "slack_voltage", "selected_bus_ids", "selected_voltages", "plot_file"],
                ["`selected_bus_ids` and `selected_voltages` should again represent the lowest-voltage buses in ascending voltage order."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, plot_file)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "baseline_low_rank_report",
            ["selected_bus_ids", "selected_voltages"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"top_k": top_k},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "add_load_slack_threshold_report",
            ["added_load_idx", "slack_bus", "slack_voltage", "threshold", "selected_bus_ids", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "slack_plot_low_rank_report",
            ["slack_setpoint", "slack_voltage", "selected_bus_ids", "selected_voltages", "plot_file"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            plot_filename=plot_file,
            task_params={"top_k": top_k},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "low_buses_add_slack_plot",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_line_blueprint_e(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    top_k = variant["top_k"][local_index]
    scale = variant["scale_values"][local_index]
    threshold = variant["angle_threshold"][local_index]
    add_bus = variant["add_buses"][local_index]
    p0 = round(0.012 + 0.002 * local_index, 3)
    q0 = round(0.006 + 0.002 * local_index, 3)
    add_idx = f"PQ_VERIFY_{scenario_number:03d}_E"
    plot_file = f"scenario_{scenario_number:03d}_turn3_voltage.png"

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[scenario_number % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow and report the top-{top_k} lines by absolute sending-end phase angle.",
            _contract_lines(
                ["selected_line_ids", "selected_line_metrics"],
                ["`selected_line_metrics` should contain the absolute sending-end phase angles in descending order."],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_all_pq", "factor": scale}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, scale every PQ load by {scale:.3f}, rerun power flow, and report every line whose absolute sending-end phase angle is above {threshold:.3f} radians.",
            _contract_lines(
                ["scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep the scaled PQ-load change from the previous turn.",
            f"Also add one new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
            f"After rerunning, save a line plot of bus voltages to '{plot_file}' and report the maximum-voltage bus and minimum-voltage bus.",
            _contract_lines(
                ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].append(
        {"label": "line angle array", "pattern": r"Line\.a1\.e", "weight": 1.0}
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t2["current_checks"].append(
        {"label": "line angle array", "pattern": r"Line\.a1\.e", "weight": 1.0}
    )
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, plot_file)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "line_topk_report",
            ["selected_line_ids", "selected_line_metrics"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"top_k": top_k},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "scaled_line_threshold_report",
            ["scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"angle_threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "add_load_voltage_plot_report",
            ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            plot_filename=plot_file,
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "line_topk_scale_threshold_plot",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_line_blueprint_f(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    slack_value = variant["slack_values"][local_index]
    scale = variant["scale_values"][local_index]
    threshold = variant["angle_threshold"][local_index]
    top_k = variant["top_k"][local_index]

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 1) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            "Run power flow and report the maximum-voltage bus, minimum-voltage bus, and slack-bus voltage.",
            _contract_lines(
                ["slack_bus", "slack_voltage", "max_bus", "max_voltage", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_slack_v0", "value": slack_value}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same case, set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and report the top-{top_k} lines by absolute sending-end phase angle.",
            _contract_lines(
                ["slack_setpoint", "slack_voltage", "selected_line_ids", "selected_line_metrics"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "scale_all_pq", "factor": scale}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
            "Keep the slack-bus adjustment from the previous turn.",
            f"Also scale every PQ load by {scale:.3f}, rerun power flow, and report all lines whose absolute sending-end phase angle is above {threshold:.3f} radians.",
            _contract_lines(
                ["slack_setpoint", "scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t2["current_checks"].append(
        {"label": "line angle array", "pattern": r"Line\.a1\.e", "weight": 1.0}
    )
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)
    checks_t3["current_checks"].append(
        {"label": "line angle array", "pattern": r"Line\.a1\.e", "weight": 1.0}
    )

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "baseline_slack_extremes_report",
            ["slack_bus", "slack_voltage", "max_bus", "max_voltage", "min_bus", "min_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
        ),
        _make_turn(
            2,
            turn2_prompt,
            "slack_line_topk_report",
            ["slack_setpoint", "slack_voltage", "selected_line_ids", "selected_line_metrics"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"top_k": top_k},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "slack_scaled_line_threshold_report",
            ["slack_setpoint", "scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"angle_threshold": threshold},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "voltage_then_slack_line_threshold",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_case_edit_n1_blueprint_g(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pq_target_buses"][local_index]
    scale_factor = variant["scale_values"][local_index]
    candidate_lines = copy.deepcopy(variant["line_candidates"][local_index : local_index + 3])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[scenario_number % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow, locate the existing PQ load connected to bus {target_bus}, and report its device idx, its current p0 and q0, and the solved slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, locate the existing PQ load at bus {target_bus}, scale both p0 and q0 of that load by {scale_factor:.3f}, rerun power flow, and report the updated device idx, updated p0/q0, and the minimum-voltage bus.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "n1_screening", "candidate_lines": copy.deepcopy(candidate_lines)}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep the targeted PQ-load scaling from the previous turn.",
            f"Now perform an N-1 screening over these candidate lines, one outage at a time, always starting from the same modified case: {_line_pair_list_text(candidate_lines)}.",
            "For each contingency, open only that one line, rerun power flow, and identify which outage gives the lowest minimum bus voltage.",
            _contract_lines(
                ["scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_min_bus", "worst_min_voltage"],
                ["`candidate_line_ids` must list the screened line ids in the same order as the candidate bus-pair list."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "n1_screening_report",
            ["scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_min_bus", "worst_min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"candidate_lines": copy.deepcopy(candidate_lines)},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "targeted_pq_edit_then_n1_screening",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_targeted_pv_line_blueprint_h(
    variant: Dict[str, Any],
    scenario_number: int,
    local_index: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pv_target_buses"][local_index]
    pv_setpoint = variant["pv_values"][local_index]
    threshold = _high_threshold_values(variant)[local_index]
    line = copy.deepcopy(variant["line_candidates"][local_index])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 1) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow, locate the existing PV device connected to bus {target_bus}, and report its idx, its current v0 setpoint, and the solved voltage at that bus.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_pv_bus_v0", "bus": target_bus, "value": pv_setpoint}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, locate the PV device at bus {target_bus}, set its v0 target to {pv_setpoint:.3f}, rerun power flow, and report the updated idx, setpoint, solved PV-bus voltage, and how many buses are above {threshold:.3f} p.u.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
            f"Keep the PV adjustment from the previous turn, then open the line between buses {line['bus1']} and {line['bus2']}, rerun power flow, and report the opened line id, the opened line bus pair, the slack-bus voltage, and the minimum-voltage bus.",
            _contract_lines(
                ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PV bus array", "pattern": r"PV\.bus\.v", "weight": 1.0},
            {"label": "PV idx array", "pattern": r"PV\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pv_bus_inspection_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pv_bus_adjust_threshold_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pv_line_outage_report",
            ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "targeted_pv_edit_then_line_outage",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_targeted_pq_outage_threshold_blueprint_i(
    variant: Dict[str, Any],
    scenario_number: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pq_target_buses"][0]
    scale_factor = variant["scale_values"][0]
    threshold = _low_threshold_values(variant)[1]
    line = copy.deepcopy(variant["line_candidates"][1])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 2) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow, locate the existing PQ load connected to bus {target_bus}, and report its device idx, its current p0 and q0, and the solved slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[scenario_number % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, locate the existing PQ load at bus {target_bus}, scale both p0 and q0 of that load by {scale_factor:.3f}, rerun power flow, and report the updated device idx, updated p0/q0, and every bus below {threshold:.3f} p.u.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order."],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            "Keep the targeted PQ-load scaling from the previous turn.",
            f"Then open the line between buses {line['bus1']} and {line['bus2']}, rerun power flow, and report the opened line id, the opened line bus pair, every bus below {threshold:.3f} p.u., and the minimum-voltage bus.",
            _contract_lines(
                ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order after the outage."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_threshold_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pq_line_outage_threshold_report",
            ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "targeted_pq_scale_then_line_outage_threshold",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_generalized_targeted_pq_trip_blueprint_j(
    variant: Dict[str, Any],
    scenario_number: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pq_target_buses"][1]
    scale_factor = variant["scale_values"][1]
    threshold = _low_threshold_values(variant)[2]
    line = copy.deepcopy(variant["line_candidates"][0])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 1) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Inspect the demand element that is already tied to bus {target_bus}.",
            "After a power-flow solution, return that record's idx together with its present p0, q0, and the slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep working from the same study state.",
            f"Find that same demand record on bus {target_bus}, increase both active and reactive demand by a factor of {scale_factor:.3f}, solve again, and then list every bus under {threshold:.3f} p.u. together with the updated device data.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order."],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
            "Keep that demand increase in place.",
            f"Now take out the branch that links buses {line['bus1']} and {line['bus2']}, solve the modified network again, and report the opened branch id, the opened bus pair, every bus below {threshold:.3f} p.u., and the minimum-voltage bus.",
            _contract_lines(
                ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order after the branch outage."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_threshold_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pq_line_outage_threshold_report",
            ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "generalized_targeted_pq_then_branch_trip",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_generalized_targeted_pv_trip_blueprint_k(
    variant: Dict[str, Any],
    scenario_number: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pv_target_buses"][1]
    pv_setpoint = variant["pv_values"][1]
    threshold = _high_threshold_values(variant)[1]
    line = copy.deepcopy(variant["line_candidates"][2])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[(scenario_number + 2) % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Inspect the generator voltage-control record tied to bus {target_bus}.",
            "After solving the case, return that record's idx, its present v0 target, and the solved voltage at that bus.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_pv_bus_v0", "bus": target_bus, "value": pv_setpoint}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            "Keep working from the same study state.",
            f"Move that same voltage-control record on bus {target_bus} to a v0 target of {pv_setpoint:.3f}, solve again, and report the updated idx, the applied setpoint, the solved PV-bus voltage, and how many buses are higher than {threshold:.3f} p.u.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
            "Keep that generator voltage-target change in place.",
            f"Now trip the branch joining buses {line['bus1']} and {line['bus2']}, solve the modified network again, and report the opened branch id, the opened bus pair, the slack-bus voltage, and the minimum-voltage bus.",
            _contract_lines(
                ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PV bus array", "pattern": r"PV\.bus\.v", "weight": 1.0},
            {"label": "PV idx array", "pattern": r"PV\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pv_bus_inspection_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pv_bus_adjust_threshold_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pv_line_outage_report",
            ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "generalized_targeted_pv_then_branch_trip",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_open_story_pq_branch_blueprint_l(
    variant: Dict[str, Any],
    scenario_id: str,
) -> Dict[str, Any]:
    uploaded_filename = None if variant["case_source"] == "builtin" else f"{scenario_id}{variant['extension']}"
    target_bus = variant["pq_target_buses"][1]
    scale_factor = variant["scale_values"][1]
    threshold = _low_threshold_values(variant)[2]
    line = copy.deepcopy(variant["line_candidates"][1])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[1],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Work out which demand record belongs to bus {target_bus}.",
            "After you solve the network, report that record's identifier, its present p0 and q0, and the slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[0],
            "Keep following the same study state.",
            f"Make that same demand record {int(round((scale_factor - 1.0) * 100))}% heavier on both active and reactive components, solve again, and list every bus under {threshold:.3f} p.u. together with the updated device data.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order."],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[3],
            "Leave that heavier demand in place.",
            f"Put the transmission corridor between buses {line['bus1']} and {line['bus2']} out of service, solve the modified case again, and report the opened branch id, the opened bus pair, every bus under {threshold:.3f} p.u., and the weakest bus.",
            _contract_lines(
                ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
                ["`selected_bus_ids` should list all buses below the threshold in ascending bus order after the outage."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_threshold_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "threshold", "selected_bus_ids", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pq_line_outage_threshold_report",
            ["scale_factor", "opened_line_id", "opened_line_bus_pair", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"threshold": threshold},
        ),
    ]
    return {
        "scenario_id": scenario_id,
        "blueprint": "open_story_pq_branch_trip",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_open_story_pv_branch_blueprint_m(
    variant: Dict[str, Any],
    scenario_id: str,
) -> Dict[str, Any]:
    uploaded_filename = None if variant["case_source"] == "builtin" else f"{scenario_id}{variant['extension']}"
    target_bus = variant["pv_target_buses"][1]
    pv_setpoint = variant["pv_values"][1]
    threshold = _high_threshold_values(variant)[1]
    line = copy.deepcopy(variant["line_candidates"][2])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[2],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Identify the generator-side voltage-control record associated with bus {target_bus}.",
            "After solving the case, return that record's idx, its present voltage target, and the solved voltage at that bus.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "set_pv_bus_v0", "bus": target_bus, "value": pv_setpoint}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[1],
            "Stay with the same modified study.",
            f"Raise that regulator target to {pv_setpoint:.3f}, solve again, and report the updated idx, the applied setpoint, the solved PV-bus voltage, and how many buses are higher than {threshold:.3f} p.u.",
            _contract_lines(
                ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
                [],
            ),
        ]
    )

    current_ops_t3 = [
        {
            "type": "line_outage_by_pair",
            "line_id": line["line_id"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
        }
    ]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[2],
            "Keep that regulator change in place.",
            f"Now knock the {line['bus1']}-{line['bus2']} corridor out of service, solve the network again, and report the opened branch id, the opened bus pair, the slack-bus voltage, and the minimum-voltage bus.",
            _contract_lines(
                ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PV bus array", "pattern": r"PV\.bus\.v", "weight": 1.0},
            {"label": "PV idx array", "pattern": r"PV\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pv_bus_inspection_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pv_bus_adjust_threshold_report",
            ["pv_bus", "pv_idx", "pv_setpoint", "pv_voltage", "threshold", "selected_count"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus, "threshold": threshold},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "pv_line_outage_report",
            ["pv_setpoint", "opened_line_id", "opened_line_bus_pair", "slack_bus", "slack_voltage", "min_bus", "min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
        ),
    ]
    return {
        "scenario_id": scenario_id,
        "blueprint": "open_story_pv_branch_trip",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_open_story_n1_blueprint_n(
    variant: Dict[str, Any],
    scenario_id: str,
) -> Dict[str, Any]:
    uploaded_filename = None if variant["case_source"] == "builtin" else f"{scenario_id}{variant['extension']}"
    target_bus = variant["pq_target_buses"][0]
    scale_factor = variant["scale_values"][0]
    candidate_lines = copy.deepcopy(variant["line_candidates"][0:3])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[0],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Figure out which demand record sits on bus {target_bus}.",
            "After you solve the system, report that record's identifier together with its present p0, q0, and the slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[0],
            "Stay with that same demand record.",
            f"Make both demand components {int(round((scale_factor - 1.0) * 100))}% higher, solve again, and report the updated identifier, updated p0/q0, and the minimum-voltage bus.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "n1_screening", "candidate_lines": copy.deepcopy(candidate_lines)}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[2],
            "Keep that heavier demand in place.",
            f"Then screen this outage set one at a time, always restarting from the same stressed case: {_line_pair_list_text(candidate_lines)}.",
            "Tell me which single outage produces the weakest voltage floor and report the screened branch ids in the same order as the outage set.",
            _contract_lines(
                ["scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_min_bus", "worst_min_voltage"],
                ["`candidate_line_ids` must list the screened line ids in the same order as the outage set."],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "n1_screening_report",
            ["scale_factor", "candidate_line_ids", "worst_line_id", "worst_line_bus_pair", "worst_min_bus", "worst_min_voltage"],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"candidate_lines": copy.deepcopy(candidate_lines)},
        ),
    ]
    return {
        "scenario_id": scenario_id,
        "blueprint": "open_story_targeted_n1_screening",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_failure_aware_n1_blueprint_o(
    variant: Dict[str, Any],
    scenario_number: int,
) -> Dict[str, Any]:
    uploaded_filename = _uploaded_filename(variant, scenario_number)
    target_bus = variant["pq_target_buses"][0]
    scale_factor = variant["scale_values"][0]
    candidate_lines = copy.deepcopy(variant.get("stress_line_candidates", variant["line_candidates"])[:3])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[scenario_number % len(CODE_ONLY_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Run power flow, locate the existing PQ load connected to bus {target_bus}, and report its device idx, its current p0 and q0, and the solved slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
            _case_phrase(variant, uploaded_filename, explicit=False),
            f"Keep the same study, scale both p0 and q0 of the existing PQ load at bus {target_bus} by {scale_factor:.3f}, rerun power flow, and report the updated idx, updated p0/q0, and the minimum-voltage bus.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "n1_screening", "candidate_lines": copy.deepcopy(candidate_lines)}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
            "Keep the targeted PQ-load scaling from the previous turn.",
            f"Now perform a failure-aware N-1 screening over these candidate lines, one outage at a time, always restarting from the same modified case: {_line_pair_list_text(candidate_lines)}.",
            "After each outage, inspect whether power flow converged and also inspect exit_code, island count, no-slack islands, and islanded bus count before trusting the voltages.",
            "Treat any outage with a no-slack island or a non-converged power flow as more severe than a converged outage. Among outages with the same status class, choose the one with the lowest minimum bus voltage.",
            _contract_lines(
                [
                    "scale_factor",
                    "candidate_line_ids",
                    "worst_line_id",
                    "worst_line_bus_pair",
                    "worst_outage_status",
                    "worst_exit_code",
                    "worst_island_count",
                    "worst_no_slack_islands",
                    "worst_islanded_bus_count",
                    "worst_min_bus",
                    "worst_min_voltage",
                ],
                [
                    "`candidate_line_ids` must list the screened line ids in the same order as the candidate bus-pair list.",
                    "`worst_outage_status` must be one of `converged`, `converged_with_islanding`, `not_converged`, or `no_slack_island`.",
                ],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)
    checks_t3["current_checks"].extend(
        [
            {"label": "power flow converged", "pattern": r"PFlow\.converged|bool\(\s*ssa\.PFlow\.run\(", "weight": 1.0},
            {"label": "exit code inspection", "pattern": r"exit_code", "weight": 1.0},
            {"label": "island set inspection", "pattern": r"(?:Bus\.island_sets|getattr\(\s*ssa\.Bus\s*,\s*[\"']island_sets[\"'])", "weight": 1.0},
            {"label": "no-slack island inspection", "pattern": r"(?:Bus\.nosw_island|getattr\(\s*ssa\.Bus\s*,\s*[\"']nosw_island[\"'])", "weight": 1.0},
        ]
    )

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "n1_failure_aware_screening_report",
            [
                "scale_factor",
                "candidate_line_ids",
                "worst_line_id",
                "worst_line_bus_pair",
                "worst_outage_status",
                "worst_exit_code",
                "worst_island_count",
                "worst_no_slack_islands",
                "worst_islanded_bus_count",
                "worst_min_bus",
                "worst_min_voltage",
            ],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"candidate_lines": copy.deepcopy(candidate_lines)},
        ),
    ]
    return {
        "scenario_id": f"scenario_{scenario_number:03d}",
        "blueprint": "failure_aware_targeted_pq_then_n1_screening",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def _build_open_story_failure_aware_n1_blueprint_o(
    variant: Dict[str, Any],
    scenario_id: str,
) -> Dict[str, Any]:
    uploaded_filename = None if variant["case_source"] == "builtin" else f"{scenario_id}{variant['extension']}"
    target_bus = variant["pq_target_buses"][0]
    scale_factor = variant["scale_values"][0]
    candidate_lines = copy.deepcopy(variant.get("stress_line_candidates", variant["line_candidates"])[:3])

    turn1_prompt = "\n".join(
        [
            CODE_ONLY_OPENERS[1],
            _case_phrase(variant, uploaded_filename, explicit=True),
            f"Find the demand record on bus {target_bus}, solve the case, and report that record's identifier together with its present p0, q0, and the slack-bus voltage.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
                [],
            ),
        ]
    )

    current_ops_t2 = [{"type": "scale_pq_at_bus", "bus": target_bus, "factor": scale_factor}]
    turn2_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[1],
            "Stay with that same demand record.",
            f"Make both demand components {int(round((scale_factor - 1.0) * 100))}% higher, solve again, and report the updated identifier, updated p0/q0, and the minimum-voltage bus.",
            _contract_lines(
                ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
                [],
            ),
        ]
    )

    current_ops_t3 = [{"type": "n1_screening", "candidate_lines": copy.deepcopy(candidate_lines)}]
    turn3_prompt = "\n".join(
        [
            FOLLOW_UP_OPENERS[3],
            "Keep that heavier demand in place.",
            f"Then screen this stressed outage set one at a time, always restarting from the same stressed case: {_line_pair_list_text(candidate_lines)}.",
            "For each outage, inspect convergence, exit_code, island count, no-slack islands, and islanded bus count before trusting the voltages.",
            "Treat any outage with a no-slack island or a non-converged power flow as more severe than a converged outage. Among outages with the same status class, choose the one with the lowest minimum bus voltage.",
            _contract_lines(
                [
                    "scale_factor",
                    "candidate_line_ids",
                    "worst_line_id",
                    "worst_line_bus_pair",
                    "worst_outage_status",
                    "worst_exit_code",
                    "worst_island_count",
                    "worst_no_slack_islands",
                    "worst_islanded_bus_count",
                    "worst_min_bus",
                    "worst_min_voltage",
                ],
                [
                    "`candidate_line_ids` must list the screened line ids in the same order as the outage set.",
                    "`worst_outage_status` must be one of `converged`, `converged_with_islanding`, `not_converged`, or `no_slack_island`.",
                ],
            ),
        ]
    )

    checks_t1 = _build_common_checks(variant, uploaded_filename, [], [], None)
    checks_t1["current_checks"].extend(
        [
            {"label": "PQ bus array", "pattern": r"PQ\.bus\.v", "weight": 1.0},
            {"label": "PQ idx array", "pattern": r"PQ\.idx\.v", "weight": 1.0},
        ]
    )
    checks_t2 = _build_common_checks(variant, uploaded_filename, current_ops_t2, [], None)
    checks_t3 = _build_common_checks(variant, uploaded_filename, current_ops_t3, current_ops_t2, None)
    checks_t3["current_checks"].extend(
        [
            {"label": "power flow converged", "pattern": r"PFlow\.converged|bool\(\s*ssa\.PFlow\.run\(", "weight": 1.0},
            {"label": "exit code inspection", "pattern": r"exit_code", "weight": 1.0},
            {"label": "island set inspection", "pattern": r"(?:Bus\.island_sets|getattr\(\s*ssa\.Bus\s*,\s*[\"']island_sets[\"'])", "weight": 1.0},
            {"label": "no-slack island inspection", "pattern": r"(?:Bus\.nosw_island|getattr\(\s*ssa\.Bus\s*,\s*[\"']nosw_island[\"'])", "weight": 1.0},
        ]
    )

    turns = [
        _make_turn(
            1,
            turn1_prompt,
            "pq_bus_inspection_report",
            ["target_pq_bus", "target_pq_idx", "target_p0", "target_q0", "slack_bus", "slack_voltage"],
            [],
            [],
            checks_t1["current_checks"],
            checks_t1["carry_checks"],
            checks_t1["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            2,
            turn2_prompt,
            "pq_bus_scale_report",
            ["target_pq_bus", "target_pq_idx", "scale_factor", "target_p0", "target_q0", "min_bus", "min_voltage"],
            current_ops_t2,
            current_ops_t2,
            checks_t2["current_checks"],
            checks_t2["carry_checks"],
            checks_t2["forbidden_patterns"],
            task_params={"target_bus": target_bus},
        ),
        _make_turn(
            3,
            turn3_prompt,
            "n1_failure_aware_screening_report",
            [
                "scale_factor",
                "candidate_line_ids",
                "worst_line_id",
                "worst_line_bus_pair",
                "worst_outage_status",
                "worst_exit_code",
                "worst_island_count",
                "worst_no_slack_islands",
                "worst_islanded_bus_count",
                "worst_min_bus",
                "worst_min_voltage",
            ],
            current_ops_t3,
            current_ops_t2 + current_ops_t3,
            checks_t3["current_checks"],
            checks_t3["carry_checks"],
            checks_t3["forbidden_patterns"],
            task_params={"candidate_lines": copy.deepcopy(candidate_lines)},
        ),
    ]
    return {
        "scenario_id": scenario_id,
        "blueprint": "open_story_failure_aware_n1_screening",
        "case_family": variant["case_family"],
        "case_source": variant["case_source"],
        "source_case_path": variant["source_case_path"],
        "uploaded_filename": uploaded_filename,
        "turns": turns,
    }


def build_verification_suite(total_scenarios: int = FULL_SUITE_SCENARIO_COUNT) -> List[Dict[str, Any]]:
    variants = _build_case_variants()
    voltage_variants = variants[:6]
    line_variants = variants[6:]
    variants_by_id = {variant["variant_id"]: variant for variant in variants}

    scenarios: List[Dict[str, Any]] = []
    scenario_number = 1

    for variant in voltage_variants:
        for local_index in range(4):
            scenarios.append(_build_voltage_blueprint_a(variant, scenario_number, local_index))
            scenario_number += 1
        for local_index in range(4):
            scenarios.append(_build_voltage_blueprint_b(variant, scenario_number, local_index))
            scenario_number += 1
        for local_index in range(3):
            scenarios.append(_build_voltage_blueprint_c(variant, scenario_number, local_index))
            scenario_number += 1
        for local_index in range(3):
            scenarios.append(_build_voltage_blueprint_d(variant, scenario_number, local_index))
            scenario_number += 1

    for variant in line_variants:
        for local_index in range(4):
            scenarios.append(_build_line_blueprint_e(variant, scenario_number, local_index))
            scenario_number += 1
        for local_index in range(4):
            scenarios.append(_build_line_blueprint_f(variant, scenario_number, local_index))
            scenario_number += 1

    for variant in variants:
        for local_index in range(2):
            scenarios.append(_build_case_edit_n1_blueprint_g(variant, scenario_number, local_index))
            scenario_number += 1
        for local_index in range(2):
            scenarios.append(_build_targeted_pv_line_blueprint_h(variant, scenario_number, local_index))
            scenario_number += 1
        scenarios.append(_build_targeted_pq_outage_threshold_blueprint_i(variant, scenario_number))
        scenario_number += 1
        scenarios.append(_build_failure_aware_n1_blueprint_o(variant, scenario_number))
        scenario_number += 1

    for variant in voltage_variants:
        scenarios.append(_build_generalized_targeted_pq_trip_blueprint_j(variant, scenario_number))
        scenario_number += 1
        scenarios.append(_build_generalized_targeted_pv_trip_blueprint_k(variant, scenario_number))
        scenario_number += 1

    expansion_scenarios = [
        _build_open_story_pq_branch_blueprint_l(
            variants_by_id["builtin_ieee39"],
            f"scenario_{scenario_number:03d}",
        ),
        _build_open_story_pv_branch_blueprint_m(
            variants_by_id["uploaded_ieee14"],
            f"scenario_{scenario_number + 1:03d}",
        ),
        _build_open_story_n1_blueprint_n(
            variants_by_id["builtin_ieee14"],
            f"scenario_{scenario_number + 2:03d}",
        ),
        _build_open_story_failure_aware_n1_blueprint_o(
            variants_by_id["uploaded_kundur"],
            f"scenario_{scenario_number + 3:03d}",
        ),
    ]
    scenarios.extend(expansion_scenarios)
    scenario_number += len(expansion_scenarios)

    if len(scenarios) != FULL_SUITE_SCENARIO_COUNT:
        raise ValueError(
            f"Expected {FULL_SUITE_SCENARIO_COUNT} scenarios, built {len(scenarios)} instead."
        )

    if total_scenarios <= 0:
        return []
    return copy.deepcopy(scenarios[:total_scenarios])


def build_open_generalization_suite() -> List[Dict[str, Any]]:
    variants = {variant["variant_id"]: variant for variant in _build_case_variants()}
    scenarios = [
        _build_open_story_pq_branch_blueprint_l(variants["builtin_ieee14"], "open_scenario_001"),
        _build_open_story_pq_branch_blueprint_l(variants["uploaded_ieee39"], "open_scenario_002"),
        _build_open_story_pv_branch_blueprint_m(variants["builtin_kundur"], "open_scenario_003"),
        _build_open_story_n1_blueprint_n(variants["uploaded_ieee14"], "open_scenario_004"),
        _build_open_story_failure_aware_n1_blueprint_o(variants["builtin_ieee39"], "open_scenario_005"),
    ]
    if len(scenarios) != OPEN_GENERALIZATION_SCENARIO_COUNT:
        raise ValueError(
            f"Expected {OPEN_GENERALIZATION_SCENARIO_COUNT} open scenarios, built {len(scenarios)} instead."
        )
    return copy.deepcopy(scenarios)
