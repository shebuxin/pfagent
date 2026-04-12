#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
# knowledge/finetuning/scripts/ -> knowledge/finetuning/ -> knowledge/ -> repo root
REPO_ROOT = SCRIPTS_DIR.parent.parent.parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.oracle import compute_oracle_turn_result  # noqa: E402

try:  # pragma: no cover - script-style import
    from generate_verified_finetune_examples import CASES, CaseSpec
except ModuleNotFoundError:  # pragma: no cover - package-style import fallback
    from knowledge.finetuning.scripts.generate_verified_finetune_examples import CASES, CaseSpec


OUTPUT_JSON = DATA_DIR / "generalized_verified_training_examples.json"
REPORT_JSON = DATA_DIR / "generalized_verified_training_examples.report.json"


FIRST_TURN_OPENERS = (
    "Return one complete runnable Python script only, inside a single ```python``` block.",
    "Answer with one runnable Python script only and no explanation.",
    "Give me a single full Python script in one fenced code block and nothing else.",
    "Please reply with one runnable Python script only.",
)

FOLLOW_UP_OPENERS = (
    "Follow-up: keep the existing study context and send a brand-new complete script.",
    "Next step: rebuild the whole script for the updated study, code only.",
    "Please revise the prior study as a fresh full script and return code only.",
    "Keep the conversation context and answer with one new runnable script only.",
)


@dataclass
class ConversationTurn:
    turn_id: int
    user: str
    assistant: str
    expected_stdout: str
    artifact_files: List[str] = field(default_factory=list)


@dataclass
class ConversationScenario:
    scenario_id: str
    family: str
    case_key: str
    source_case_path: str
    uploaded_filename: Optional[str]
    turns: List[ConversationTurn]

    @property
    def uploaded_files(self) -> Dict[str, str]:
        if not self.uploaded_filename:
            return {}
        return {self.uploaded_filename: self.source_case_path}

    def to_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _json_stdout(payload: Dict[str, Any]) -> str:
    return "RESULT_JSON=" + json.dumps(payload, sort_keys=True)


def _normalize_stdout(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _make_uploaded_filename(case_spec: CaseSpec, scenario_number: int) -> Optional[str]:
    if not case_spec.is_uploaded:
        return None
    suffix = Path(case_spec.source_case).suffix
    return f"generalized_{case_spec.key}_{scenario_number:03d}{suffix}"


def _case_phrase(case_spec: CaseSpec, uploaded_filename: Optional[str], explicit: bool) -> str:
    if case_spec.is_uploaded:
        if explicit:
            return f"Use my uploaded case file '{uploaded_filename}' from the current working directory."
        return "Keep using the same uploaded case file from the earlier turn."
    if explicit:
        return f"Use {case_spec.prompt_label}."
    return "Keep using the same built-in ANDES case from the earlier turn."


def _contract_lines(keys: Sequence[str], extra: Sequence[str]) -> str:
    lines = [
        "End the script by printing exactly one line that begins with RESULT_JSON=",
        f"RESULT_JSON must contain these keys: {', '.join(keys)}.",
        "Use plain Python ints, floats, strings, and lists in RESULT_JSON.",
        "Round every floating-point value in RESULT_JSON to 6 decimals.",
    ]
    lines.extend(extra)
    return "\n".join(f"- {line}" for line in lines)


def _latest_op(operations: Sequence[Dict[str, Any]], op_type: str) -> Dict[str, Any]:
    for op in reversed(operations):
        if op["type"] == op_type:
            return op
    raise KeyError(f"Missing operation {op_type}")


def _render_case_loader(case_spec: CaseSpec, uploaded_filename: Optional[str]) -> List[str]:
    if uploaded_filename:
        return [
            "case_path = os.path.join(os.getcwd(), %r)" % uploaded_filename,
            "ssa = andes.load(case_path, setup=False, no_output=True)",
        ]
    return [
        "ssa = andes.load(",
        f"    andes.get_case({case_spec.source_case!r}),",
        "    setup=False,",
        "    no_output=True,",
        ")",
    ]


def _render_pre_setup_ops(operations: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for op in operations:
        if op["type"] != "add_pq":
            continue
        lines.extend(
            [
                "ssa.add(",
                "    'PQ',",
                "    param_dict={",
                f"        'bus': {op['bus']},",
                f"        'idx': {op['idx']!r},",
                f"        'p0': {op['p0']},",
                f"        'q0': {op['q0']},",
                "    },",
                ")",
            ]
        )
    return lines


def _render_post_setup_ops(operations: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for op in operations:
        if op["type"] == "scale_all_pq":
            lines.extend(
                [
                    f"scale_factor = {_fmt(op['factor'])}",
                    "ssa.PQ.set(src='p0', idx=ssa.PQ.idx.v, attr='v', value=scale_factor * ssa.PQ.p0.v)",
                    "ssa.PQ.set(src='q0', idx=ssa.PQ.idx.v, attr='v', value=scale_factor * ssa.PQ.q0.v)",
                ]
            )
        elif op["type"] == "set_slack_v0":
            lines.extend(
                [
                    f"slack_setpoint = {_fmt(op['value'])}",
                    "ssa.Slack.set(src='v0', idx=[ssa.Slack.idx.v[0]], attr='v', value=[slack_setpoint])",
                ]
            )
        elif op["type"] == "set_first_pv_v0":
            lines.extend(
                [
                    f"pv_setpoint = {_fmt(op['value'])}",
                    "ssa.PV.set(src='v0', idx=[ssa.PV.idx.v[0]], attr='v', value=[pv_setpoint])",
                ]
            )
    return lines


def _render_plot_lines(plot_filename: str, style: str) -> List[str]:
    lines = [
        f"plot_file = {plot_filename!r}",
        "plt.figure(figsize=(10, 4))",
    ]
    if style == "bar":
        lines.extend(
            [
                "plt.bar(bus_ids.astype(str), bus_v)",
                "plt.xticks(rotation=90)",
            ]
        )
    else:
        lines.extend(
            [
                "plt.plot(bus_ids, bus_v, marker='o', linewidth=1.5)",
                "plt.xticks(rotation=90)",
            ]
        )
    lines.extend(
        [
            "plt.xlabel('Bus')",
            "plt.ylabel('Voltage (p.u.)')",
            "plt.tight_layout()",
            "plt.savefig(plot_file, dpi=150)",
            "plt.close()",
        ]
    )
    return lines


def _render_report_lines(
    report_kind: str,
    result_keys: Sequence[str],
    task_params: Dict[str, Any],
    cumulative_operations: Sequence[Dict[str, Any]],
    plot_filename: Optional[str],
) -> List[str]:
    lines: List[str] = []
    need_bus_arrays = report_kind not in {
        "line_topk_report",
        "scaled_line_threshold_report",
    }
    if need_bus_arrays:
        lines.extend(
            [
                "bus_ids = np.asarray(ssa.Bus.idx.v)",
                "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
            ]
        )

    if report_kind in {
        "baseline_high_rank_report",
        "slack_adjust_report",
        "add_load_slack_threshold_report",
        "slack_plot_low_rank_report",
        "baseline_slack_extremes_report",
        "slack_line_topk_report",
    }:
        lines.extend(
            [
                "slack_bus = int(ssa.Slack.bus.v[0])",
                "slack_index = int(np.where(bus_ids == slack_bus)[0][0])",
                "slack_voltage = _round_float(bus_v[slack_index])",
            ]
        )

    if report_kind == "baseline_high_rank_report":
        top_k = int(task_params["top_k"])
        lines.extend(
            [
                f"top_k = {top_k}",
                "rank_indices = np.argsort(bus_v)[-top_k:][::-1]",
                "result = {",
                "    'slack_bus': slack_bus,",
                "    'slack_voltage': slack_voltage,",
                "    'selected_bus_ids': [int(bus_ids[i]) for i in rank_indices],",
                "    'selected_voltages': [_round_float(bus_v[i]) for i in rank_indices],",
                "}",
            ]
        )
    elif report_kind == "add_load_threshold_report":
        add_op = _latest_op(cumulative_operations, "add_pq")
        threshold = float(task_params["threshold"])
        lines.extend(
            [
                f"threshold = {_fmt(threshold)}",
                "mask = bus_v < threshold",
                "min_index = int(np.argmin(bus_v))",
                "result = {",
                f"    'added_load_idx': {add_op['idx']!r},",
                f"    'added_load_bus': {int(add_op['bus'])},",
                "    'threshold': _round_float(threshold),",
                "    'selected_bus_ids': [int(value) for value in bus_ids[mask]],",
                "    'selected_count': int(np.sum(mask)),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "}",
            ]
        )
    elif report_kind == "scaled_plot_report":
        scale_op = _latest_op(cumulative_operations, "scale_all_pq")
        lines.extend(_render_plot_lines(plot_filename or "generalized_plot.png", style="line"))
        lines.extend(
            [
                "max_index = int(np.argmax(bus_v))",
                "min_index = int(np.argmin(bus_v))",
                "result = {",
                f"    'scale_factor': _round_float({_fmt(scale_op['factor'])}),",
                "    'max_bus': int(bus_ids[max_index]),",
                "    'max_voltage': _round_float(bus_v[max_index]),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "    'plot_file': plot_file,",
                "}",
            ]
        )
    elif report_kind == "baseline_threshold_low_rank_report":
        threshold = float(task_params["threshold"])
        lines.extend(
            [
                f"threshold = {_fmt(threshold)}",
                "mask = bus_v > threshold",
                "low_rank = np.argsort(bus_v)[:2]",
                "result = {",
                "    'threshold': _round_float(threshold),",
                "    'selected_bus_ids': [int(value) for value in bus_ids[mask]],",
                "    'selected_count': int(np.sum(mask)),",
                "    'lowest_bus_ids': [int(bus_ids[i]) for i in low_rank],",
                "    'lowest_voltages': [_round_float(bus_v[i]) for i in low_rank],",
                "}",
            ]
        )
    elif report_kind == "slack_adjust_report":
        slack_op = _latest_op(cumulative_operations, "set_slack_v0")
        threshold = float(task_params["threshold"])
        lines.extend(
            [
                f"threshold = {_fmt(threshold)}",
                "result = {",
                "    'slack_bus': slack_bus,",
                f"    'slack_setpoint': _round_float({_fmt(slack_op['value'])}),",
                "    'slack_voltage': slack_voltage,",
                "    'selected_count': int(np.sum(bus_v < threshold)),",
                "}",
            ]
        )
    elif report_kind == "extremes_report":
        max_index = "max_index"
        min_index = "min_index"
        lines.extend(
            [
                f"{max_index} = int(np.argmax(bus_v))",
                f"{min_index} = int(np.argmin(bus_v))",
                "result = {",
                "    'max_bus': int(bus_ids[max_index]),",
                "    'max_voltage': _round_float(bus_v[max_index]),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "}",
            ]
        )
        if "added_load_idx" in result_keys:
            add_op = _latest_op(cumulative_operations, "add_pq")
            lines.append(f"result['added_load_idx'] = {add_op['idx']!r}")
        if "total_pq_count" in result_keys:
            lines.append("result['total_pq_count'] = int(len(ssa.PQ.idx.v))")
    elif report_kind == "pv_adjust_report":
        pv_op = _latest_op(cumulative_operations, "set_first_pv_v0")
        threshold = float(task_params["threshold"])
        lines.extend(
            [
                "pv_bus = int(ssa.PV.bus.v[0])",
                "pv_index = int(np.where(bus_ids == pv_bus)[0][0])",
                f"threshold = {_fmt(threshold)}",
                "result = {",
                "    'pv_bus': pv_bus,",
                f"    'pv_setpoint': _round_float({_fmt(pv_op['value'])}),",
                "    'pv_voltage': _round_float(bus_v[pv_index]),",
                "    'selected_count': int(np.sum(bus_v > threshold)),",
                "}",
            ]
        )
    elif report_kind == "scaled_bar_plot_report":
        scale_op = _latest_op(cumulative_operations, "scale_all_pq")
        lines.extend(_render_plot_lines(plot_filename or "generalized_bar.png", style="bar"))
        lines.extend(
            [
                "min_index = int(np.argmin(bus_v))",
                "max_index = int(np.argmax(bus_v))",
                "result = {",
                f"    'scale_factor': _round_float({_fmt(scale_op['factor'])}),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "    'max_bus': int(bus_ids[max_index]),",
                "    'max_voltage': _round_float(bus_v[max_index]),",
                "    'plot_file': plot_file,",
                "}",
            ]
        )
    elif report_kind == "baseline_low_rank_report":
        top_k = int(task_params["top_k"])
        lines.extend(
            [
                f"top_k = {top_k}",
                "rank_indices = np.argsort(bus_v)[:top_k]",
                "result = {",
                "    'selected_bus_ids': [int(bus_ids[i]) for i in rank_indices],",
                "    'selected_voltages': [_round_float(bus_v[i]) for i in rank_indices],",
                "}",
            ]
        )
    elif report_kind == "add_load_slack_threshold_report":
        add_op = _latest_op(cumulative_operations, "add_pq")
        threshold = float(task_params["threshold"])
        lines.extend(
            [
                f"threshold = {_fmt(threshold)}",
                "mask = bus_v < threshold",
                "result = {",
                f"    'added_load_idx': {add_op['idx']!r},",
                "    'slack_bus': slack_bus,",
                "    'slack_voltage': slack_voltage,",
                "    'threshold': _round_float(threshold),",
                "    'selected_bus_ids': [int(value) for value in bus_ids[mask]],",
                "    'selected_count': int(np.sum(mask)),",
                "}",
            ]
        )
    elif report_kind == "slack_plot_low_rank_report":
        slack_op = _latest_op(cumulative_operations, "set_slack_v0")
        top_k = int(task_params["top_k"])
        lines.extend(_render_plot_lines(plot_filename or "generalized_low_rank.png", style="line"))
        lines.extend(
            [
                f"top_k = {top_k}",
                "rank_indices = np.argsort(bus_v)[:top_k]",
                "result = {",
                f"    'slack_setpoint': _round_float({_fmt(slack_op['value'])}),",
                "    'slack_voltage': slack_voltage,",
                "    'selected_bus_ids': [int(bus_ids[i]) for i in rank_indices],",
                "    'selected_voltages': [_round_float(bus_v[i]) for i in rank_indices],",
                "    'plot_file': plot_file,",
                "}",
            ]
        )
    elif report_kind == "line_topk_report":
        top_k = int(task_params["top_k"])
        lines.extend(
            [
                "line_ids = np.asarray(ssa.Line.idx.v)",
                "abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
                f"top_k = {top_k}",
                "rank_indices = np.argsort(abs_a1)[-top_k:][::-1]",
                "result = {",
                "    'selected_line_ids': [str(line_ids[i]) for i in rank_indices],",
                "    'selected_line_metrics': [_round_float(abs_a1[i]) for i in rank_indices],",
                "}",
            ]
        )
    elif report_kind == "scaled_line_threshold_report":
        scale_op = _latest_op(cumulative_operations, "scale_all_pq")
        threshold = float(task_params["angle_threshold"])
        lines.extend(
            [
                "line_ids = np.asarray(ssa.Line.idx.v)",
                "abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
                f"angle_threshold = {_fmt(threshold)}",
                "mask = abs_a1 > angle_threshold",
                "result = {",
                f"    'scale_factor': _round_float({_fmt(scale_op['factor'])}),",
                "    'angle_threshold': _round_float(angle_threshold),",
                "    'selected_line_ids': [str(value) for value in line_ids[mask]],",
                "    'selected_count': int(np.sum(mask)),",
                "}",
            ]
        )
    elif report_kind == "add_load_voltage_plot_report":
        add_op = _latest_op(cumulative_operations, "add_pq")
        lines.extend(_render_plot_lines(plot_filename or "generalized_voltage.png", style="line"))
        lines.extend(
            [
                "max_index = int(np.argmax(bus_v))",
                "min_index = int(np.argmin(bus_v))",
                "result = {",
                f"    'added_load_idx': {add_op['idx']!r},",
                "    'max_bus': int(bus_ids[max_index]),",
                "    'max_voltage': _round_float(bus_v[max_index]),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "    'plot_file': plot_file,",
                "}",
            ]
        )
    elif report_kind == "baseline_slack_extremes_report":
        lines.extend(
            [
                "max_index = int(np.argmax(bus_v))",
                "min_index = int(np.argmin(bus_v))",
                "result = {",
                "    'slack_bus': slack_bus,",
                "    'slack_voltage': slack_voltage,",
                "    'max_bus': int(bus_ids[max_index]),",
                "    'max_voltage': _round_float(bus_v[max_index]),",
                "    'min_bus': int(bus_ids[min_index]),",
                "    'min_voltage': _round_float(bus_v[min_index]),",
                "}",
            ]
        )
    elif report_kind == "slack_line_topk_report":
        slack_op = _latest_op(cumulative_operations, "set_slack_v0")
        top_k = int(task_params["top_k"])
        lines.extend(
            [
                "line_ids = np.asarray(ssa.Line.idx.v)",
                "abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
                f"top_k = {top_k}",
                "rank_indices = np.argsort(abs_a1)[-top_k:][::-1]",
                "result = {",
                f"    'slack_setpoint': _round_float({_fmt(slack_op['value'])}),",
                "    'slack_voltage': slack_voltage,",
                "    'selected_line_ids': [str(line_ids[i]) for i in rank_indices],",
                "    'selected_line_metrics': [_round_float(abs_a1[i]) for i in rank_indices],",
                "}",
            ]
        )
    elif report_kind == "slack_scaled_line_threshold_report":
        slack_op = _latest_op(cumulative_operations, "set_slack_v0")
        scale_op = _latest_op(cumulative_operations, "scale_all_pq")
        threshold = float(task_params["angle_threshold"])
        lines.extend(
            [
                "line_ids = np.asarray(ssa.Line.idx.v)",
                "abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
                f"angle_threshold = {_fmt(threshold)}",
                "mask = abs_a1 > angle_threshold",
                "result = {",
                f"    'slack_setpoint': _round_float({_fmt(slack_op['value'])}),",
                f"    'scale_factor': _round_float({_fmt(scale_op['factor'])}),",
                "    'angle_threshold': _round_float(angle_threshold),",
                "    'selected_line_ids': [str(value) for value in line_ids[mask]],",
                "    'selected_count': int(np.sum(mask)),",
                "}",
            ]
        )
    else:  # pragma: no cover - guarded by scenario construction
        raise ValueError(f"Unsupported report kind: {report_kind}")

    lines.append('print("RESULT_JSON=" + json.dumps(result, sort_keys=True))')
    return lines


def _render_turn_code(
    case_spec: CaseSpec,
    uploaded_filename: Optional[str],
    cumulative_operations: Sequence[Dict[str, Any]],
    report_kind: str,
    result_keys: Sequence[str],
    task_params: Dict[str, Any],
    plot_filename: Optional[str],
) -> str:
    imports = ["import json", "import andes", "import numpy as np"]
    if uploaded_filename:
        imports.append("import os")
    if plot_filename:
        imports.append("import matplotlib.pyplot as plt")

    sections: List[str] = []
    sections.append("\n".join(imports))
    sections.append(
        "\n".join(
            [
                "def _round_float(value):",
                "    return round(float(value), 6)",
            ]
        )
    )

    body_lines: List[str] = []
    body_lines.extend(_render_case_loader(case_spec, uploaded_filename))
    pre_setup_ops = _render_pre_setup_ops(cumulative_operations)
    if pre_setup_ops:
        body_lines.append("")
        body_lines.extend(pre_setup_ops)
    body_lines.append("")
    body_lines.append("ssa.setup()")
    post_setup_ops = _render_post_setup_ops(cumulative_operations)
    if post_setup_ops:
        body_lines.extend(post_setup_ops)
    body_lines.append("ssa.PFlow.run()")
    body_lines.append("")
    body_lines.extend(_render_report_lines(report_kind, result_keys, task_params, cumulative_operations, plot_filename))
    sections.append("\n".join(body_lines))
    return "\n\n".join(section for section in sections if section.strip()).strip()


def _build_turn(
    case_spec: CaseSpec,
    uploaded_filename: Optional[str],
    turn_id: int,
    user: str,
    cumulative_operations: Sequence[Dict[str, Any]],
    report_kind: str,
    result_keys: Sequence[str],
    task_params: Dict[str, Any],
    plot_filename: Optional[str] = None,
) -> ConversationTurn:
    oracle_scenario = {"source_case_path": case_spec.source_case}
    oracle_turn = {
        "cumulative_operations": list(cumulative_operations),
        "report_kind": report_kind,
        "result_keys": list(result_keys),
        "task_params": dict(task_params),
        "plot_filename": plot_filename,
    }
    oracle_result = compute_oracle_turn_result(oracle_scenario, oracle_turn)
    assistant = _render_turn_code(
        case_spec=case_spec,
        uploaded_filename=uploaded_filename,
        cumulative_operations=cumulative_operations,
        report_kind=report_kind,
        result_keys=result_keys,
        task_params=task_params,
        plot_filename=plot_filename,
    )
    return ConversationTurn(
        turn_id=turn_id,
        user=user,
        assistant=assistant,
        expected_stdout=_json_stdout(oracle_result),
        artifact_files=[plot_filename] if plot_filename else [],
    )


def _build_rank_add_scale_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    top_k: int,
    add_bus: int,
    threshold: float,
    scale: float,
    p0: float,
    q0: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)
    add_idx = f"GFT_{scenario_number:03d}_A"
    plot_file = f"generalized_{scenario_number:03d}_turn3_line.png"

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[scenario_number % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                f"Run power flow and report the slack-bus voltage together with the top-{top_k} highest-voltage buses.",
                _contract_lines(
                    ["slack_bus", "slack_voltage", "selected_bus_ids", "selected_voltages"],
                    ["`selected_bus_ids` and `selected_voltages` should be in descending voltage order."],
                ),
            ]
        ),
        [],
        "baseline_high_rank_report",
        ["slack_bus", "slack_voltage", "selected_bus_ids", "selected_voltages"],
        {"top_k": top_k},
    )

    turn2_ops = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Add one new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
                f"After rerunning, report every bus below {threshold:.3f} p.u. and also identify the minimum-voltage bus.",
                _contract_lines(
                    ["added_load_idx", "added_load_bus", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
                    ["List `selected_bus_ids` in the case order returned by the solved system."],
                ),
            ]
        ),
        turn2_ops,
        "add_load_threshold_report",
        ["added_load_idx", "added_load_bus", "threshold", "selected_bus_ids", "selected_count", "min_bus", "min_voltage"],
        {"threshold": threshold},
    )

    turn3_ops = turn2_ops + [{"type": "scale_all_pq", "factor": scale}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
                "Keep the added PQ load from the previous turn.",
                f"Also scale every PQ load by a factor of {scale:.3f}, rerun power flow, and save a line plot of bus voltages to '{plot_file}'.",
                _contract_lines(
                    ["scale_factor", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
                    ["`plot_file` must exactly match the saved filename."],
                ),
            ]
        ),
        turn3_ops,
        "scaled_plot_report",
        ["scale_factor", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
        {"plot_style": "line"},
        plot_filename=plot_file,
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="rank_add_scale_plot",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def _build_threshold_slack_extremes_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    threshold_high: float,
    threshold_low: float,
    slack_value: float,
    add_bus: int,
    p0: float,
    q0: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)
    add_idx = f"GFT_{scenario_number:03d}_B"

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[(scenario_number + 1) % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                f"Run power flow, count every bus above {threshold_high:.3f} p.u., and also return the two lowest-voltage buses.",
                _contract_lines(
                    ["threshold", "selected_bus_ids", "selected_count", "lowest_bus_ids", "lowest_voltages"],
                    ["`lowest_bus_ids` must contain exactly two buses in ascending voltage order."],
                ),
            ]
        ),
        [],
        "baseline_threshold_low_rank_report",
        ["threshold", "selected_bus_ids", "selected_count", "lowest_bus_ids", "lowest_voltages"],
        {"threshold": threshold_high},
    )

    turn2_ops = [{"type": "set_slack_v0", "value": slack_value}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and report the solved slack-bus voltage plus how many buses are now below {threshold_low:.3f} p.u.",
                _contract_lines(
                    ["slack_bus", "slack_setpoint", "slack_voltage", "selected_count"],
                    [],
                ),
            ]
        ),
        turn2_ops,
        "slack_adjust_report",
        ["slack_bus", "slack_setpoint", "slack_voltage", "selected_count"],
        {"threshold": threshold_low},
    )

    turn3_ops = turn2_ops + [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
                "Keep the slack-bus change from the previous turn.",
                f"Also add one new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
                "After rerunning, report the maximum-voltage bus, minimum-voltage bus, and the total number of PQ loads now present.",
                _contract_lines(
                    ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "total_pq_count"],
                    [],
                ),
            ]
        ),
        turn3_ops,
        "extremes_report",
        ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "total_pq_count"],
        {},
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="threshold_slack_add_extremes",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def _build_lowrank_slack_plot_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    top_k: int,
    add_bus: int,
    threshold: float,
    slack_value: float,
    p0: float,
    q0: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)
    add_idx = f"GFT_{scenario_number:03d}_C"
    plot_file = f"generalized_{scenario_number:03d}_turn3_lowrank.png"

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[(scenario_number + 2) % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                f"Run power flow and report the {top_k} lowest-voltage buses.",
                _contract_lines(
                    ["selected_bus_ids", "selected_voltages"],
                    ["Return the buses in ascending voltage order."],
                ),
            ]
        ),
        [],
        "baseline_low_rank_report",
        ["selected_bus_ids", "selected_voltages"],
        {"top_k": top_k},
    )

    turn2_ops = [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[scenario_number % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Add one new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
                f"Then rerun power flow and report the solved slack-bus voltage together with every bus below {threshold:.3f} p.u.",
                _contract_lines(
                    ["added_load_idx", "slack_bus", "slack_voltage", "threshold", "selected_bus_ids", "selected_count"],
                    [],
                ),
            ]
        ),
        turn2_ops,
        "add_load_slack_threshold_report",
        ["added_load_idx", "slack_bus", "slack_voltage", "threshold", "selected_bus_ids", "selected_count"],
        {"threshold": threshold},
    )

    turn3_ops = turn2_ops + [{"type": "set_slack_v0", "value": slack_value}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
                "Keep the added PQ load from the previous turn.",
                f"Also set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and save a line plot of bus voltages to '{plot_file}'.",
                _contract_lines(
                    ["slack_setpoint", "slack_voltage", "selected_bus_ids", "selected_voltages", "plot_file"],
                    ["Again report the lowest-voltage buses in ascending voltage order."],
                ),
            ]
        ),
        turn3_ops,
        "slack_plot_low_rank_report",
        ["slack_setpoint", "slack_voltage", "selected_bus_ids", "selected_voltages", "plot_file"],
        {"top_k": top_k},
        plot_filename=plot_file,
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="lowrank_add_slack_plot",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def _build_pv_scale_bar_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    pv_value: float,
    threshold: float,
    scale: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)
    plot_file = f"generalized_{scenario_number:03d}_turn3_bar.png"

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[(scenario_number + 3) % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                "Run power flow and report the maximum-voltage bus together with the minimum-voltage bus.",
                _contract_lines(["max_bus", "max_voltage", "min_bus", "min_voltage"], []),
            ]
        ),
        [],
        "extremes_report",
        ["max_bus", "max_voltage", "min_bus", "min_voltage"],
        {},
    )

    turn2_ops = [{"type": "set_first_pv_v0", "value": pv_value}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Set the first PV voltage target to {pv_value:.3f}, rerun power flow, and report the controlled PV bus voltage plus how many buses exceed {threshold:.3f} p.u.",
                _contract_lines(["pv_bus", "pv_setpoint", "pv_voltage", "selected_count"], []),
            ]
        ),
        turn2_ops,
        "pv_adjust_report",
        ["pv_bus", "pv_setpoint", "pv_voltage", "selected_count"],
        {"threshold": threshold},
    )

    turn3_ops = turn2_ops + [{"type": "scale_all_pq", "factor": scale}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
                "Keep the PV setpoint change from the previous turn.",
                f"Also scale every PQ load by {scale:.3f}, rerun power flow, and save a bar chart of the bus voltages to '{plot_file}'.",
                _contract_lines(
                    ["scale_factor", "min_bus", "min_voltage", "max_bus", "max_voltage", "plot_file"],
                    ["Use a bar chart rather than a line chart."],
                ),
            ]
        ),
        turn3_ops,
        "scaled_bar_plot_report",
        ["scale_factor", "min_bus", "min_voltage", "max_bus", "max_voltage", "plot_file"],
        {"plot_style": "bar"},
        plot_filename=plot_file,
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="extremes_pv_scale_barplot",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def _build_line_scale_plot_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    top_k: int,
    scale: float,
    threshold: float,
    add_bus: int,
    p0: float,
    q0: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)
    add_idx = f"GFT_{scenario_number:03d}_D"
    plot_file = f"generalized_{scenario_number:03d}_turn3_voltage.png"

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[scenario_number % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                f"Run power flow and report the top-{top_k} lines by absolute sending-end phase angle.",
                _contract_lines(
                    ["selected_line_ids", "selected_line_metrics"],
                    ["Return the lines in descending order of absolute sending-end phase angle."],
                ),
            ]
        ),
        [],
        "line_topk_report",
        ["selected_line_ids", "selected_line_metrics"],
        {"top_k": top_k},
    )

    turn2_ops = [{"type": "scale_all_pq", "factor": scale}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 1) % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Scale every PQ load by {scale:.3f}, rerun power flow, and report every line whose absolute sending-end phase angle exceeds {threshold:.3f} radians.",
                _contract_lines(
                    ["scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
                    [],
                ),
            ]
        ),
        turn2_ops,
        "scaled_line_threshold_report",
        ["scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
        {"angle_threshold": threshold},
    )

    turn3_ops = turn2_ops + [{"type": "add_pq", "bus": add_bus, "idx": add_idx, "p0": p0, "q0": q0}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 2) % len(FOLLOW_UP_OPENERS)],
                "Keep the PQ scaling change from the previous turn.",
                f"Also add a new PQ load before setup at bus {add_bus} with idx '{add_idx}', p0={_fmt(p0)}, and q0={_fmt(q0)}.",
                f"Rerun power flow, save a voltage profile plot to '{plot_file}', and report the maximum-voltage and minimum-voltage buses.",
                _contract_lines(
                    ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
                    [],
                ),
            ]
        ),
        turn3_ops,
        "add_load_voltage_plot_report",
        ["added_load_idx", "max_bus", "max_voltage", "min_bus", "min_voltage", "plot_file"],
        {"plot_style": "line"},
        plot_filename=plot_file,
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="line_topk_scale_threshold_plot",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def _build_voltage_then_line_conversation(
    case_spec: CaseSpec,
    scenario_number: int,
    slack_value: float,
    top_k: int,
    scale: float,
    threshold: float,
) -> ConversationScenario:
    uploaded_filename = _make_uploaded_filename(case_spec, scenario_number)

    turn1 = _build_turn(
        case_spec,
        uploaded_filename,
        1,
        "\n".join(
            [
                FIRST_TURN_OPENERS[(scenario_number + 1) % len(FIRST_TURN_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=True),
                "Run power flow and report the slack-bus voltage together with the maximum-voltage and minimum-voltage buses.",
                _contract_lines(
                    ["slack_bus", "slack_voltage", "max_bus", "max_voltage", "min_bus", "min_voltage"],
                    [],
                ),
            ]
        ),
        [],
        "baseline_slack_extremes_report",
        ["slack_bus", "slack_voltage", "max_bus", "max_voltage", "min_bus", "min_voltage"],
        {},
    )

    turn2_ops = [{"type": "set_slack_v0", "value": slack_value}]
    turn2 = _build_turn(
        case_spec,
        uploaded_filename,
        2,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[(scenario_number + 3) % len(FOLLOW_UP_OPENERS)],
                _case_phrase(case_spec, uploaded_filename, explicit=False),
                f"Set the slack-bus voltage target to {slack_value:.3f}, rerun power flow, and report the top-{top_k} lines by absolute sending-end phase angle.",
                _contract_lines(
                    ["slack_setpoint", "slack_voltage", "selected_line_ids", "selected_line_metrics"],
                    [],
                ),
            ]
        ),
        turn2_ops,
        "slack_line_topk_report",
        ["slack_setpoint", "slack_voltage", "selected_line_ids", "selected_line_metrics"],
        {"top_k": top_k},
    )

    turn3_ops = turn2_ops + [{"type": "scale_all_pq", "factor": scale}]
    turn3 = _build_turn(
        case_spec,
        uploaded_filename,
        3,
        "\n".join(
            [
                FOLLOW_UP_OPENERS[scenario_number % len(FOLLOW_UP_OPENERS)],
                "Keep the slack-bus setpoint change from the previous turn.",
                f"Also scale every PQ load by {scale:.3f}, rerun power flow, and report every line whose absolute sending-end phase angle is above {threshold:.3f} radians.",
                _contract_lines(
                    ["slack_setpoint", "scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
                    [],
                ),
            ]
        ),
        turn3_ops,
        "slack_scaled_line_threshold_report",
        ["slack_setpoint", "scale_factor", "angle_threshold", "selected_line_ids", "selected_count"],
        {"angle_threshold": threshold},
    )

    return ConversationScenario(
        scenario_id=f"generalized_scenario_{scenario_number:03d}",
        family="voltage_then_slack_line_threshold",
        case_key=case_spec.key,
        source_case_path=case_spec.source_case,
        uploaded_filename=uploaded_filename,
        turns=[turn1, turn2, turn3],
    )


def build_scenarios() -> List[ConversationScenario]:
    scenarios: List[ConversationScenario] = []
    scenario_number = 1

    def add(scenario: ConversationScenario) -> None:
        nonlocal scenario_number
        scenarios.append(scenario)
        scenario_number += 1

    for case_key, top_k, add_bus, threshold, scale, p0, q0 in [
        ("builtin_ieee14", 4, 10, 1.014, 1.045, 0.016, 0.009),
        ("uploaded_ieee14", 5, 9, 1.012, 0.985, 0.014, 0.008),
        ("builtin_ieee39", 4, 20, 0.975, 1.030, 0.017, 0.009),
        ("uploaded_ieee39", 5, 15, 0.968, 1.045, 0.018, 0.010),
        ("builtin_gbnetwork", 6, 12, 0.955, 1.020, 0.012, 0.006),
        ("uploaded_gbnetwork", 5, 100, 0.948, 0.990, 0.010, 0.005),
        ("builtin_ei33", 4, 100007, 0.900, 1.015, 0.011, 0.006),
        ("uploaded_ei33", 5, 100029, 0.880, 0.980, 0.010, 0.005),
    ]:
        add(
            _build_rank_add_scale_conversation(
                CASES[case_key],
                scenario_number,
                top_k=top_k,
                add_bus=add_bus,
                threshold=threshold,
                scale=scale,
                p0=p0,
                q0=q0,
            )
        )

    for case_key, threshold_high, threshold_low, slack_value, add_bus, p0, q0 in [
        ("builtin_ieee14", 1.020, 1.011, 1.022, 10, 0.015, 0.009),
        ("uploaded_ieee14", 1.018, 1.010, 1.028, 5, 0.014, 0.008),
        ("builtin_ieee39", 1.038, 0.972, 1.026, 20, 0.017, 0.010),
        ("uploaded_ieee39", 1.045, 0.965, 1.018, 10, 0.018, 0.010),
        ("builtin_kundur", 0.995, 0.962, 1.008, 6, 0.020, 0.012),
        ("uploaded_kundur", 0.998, 0.958, 1.012, 9, 0.022, 0.014),
    ]:
        add(
            _build_threshold_slack_extremes_conversation(
                CASES[case_key],
                scenario_number,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                slack_value=slack_value,
                add_bus=add_bus,
                p0=p0,
                q0=q0,
            )
        )

    for case_key, top_k, add_bus, threshold, slack_value, p0, q0 in [
        ("builtin_ieee14", 4, 9, 1.012, 1.024, 0.015, 0.009),
        ("uploaded_ieee14", 3, 10, 1.013, 1.030, 0.016, 0.010),
        ("builtin_ieee39", 4, 15, 0.972, 1.020, 0.018, 0.010),
        ("uploaded_ieee39", 5, 20, 0.968, 1.032, 0.017, 0.009),
        ("builtin_kundur", 3, 4, 0.965, 1.005, 0.021, 0.013),
        ("uploaded_kundur", 4, 7, 0.960, 1.010, 0.020, 0.012),
    ]:
        add(
            _build_lowrank_slack_plot_conversation(
                CASES[case_key],
                scenario_number,
                top_k=top_k,
                add_bus=add_bus,
                threshold=threshold,
                slack_value=slack_value,
                p0=p0,
                q0=q0,
            )
        )

    for case_key, pv_value, threshold, scale in [
        ("builtin_ieee39", 1.017, 1.036, 1.035),
        ("uploaded_ieee39", 1.012, 1.030, 0.985),
    ]:
        add(
            _build_pv_scale_bar_conversation(
                CASES[case_key],
                scenario_number,
                pv_value=pv_value,
                threshold=threshold,
                scale=scale,
            )
        )

    for case_key, top_k, scale, threshold, add_bus, p0, q0 in [
        ("builtin_pjm5", 4, 1.040, 1.500, 2, 0.014, 0.008),
        ("uploaded_pjm5", 3, 0.960, 1.000, 4, 0.012, 0.007),
    ]:
        add(
            _build_line_scale_plot_conversation(
                CASES[case_key],
                scenario_number,
                top_k=top_k,
                scale=scale,
                threshold=threshold,
                add_bus=add_bus,
                p0=p0,
                q0=q0,
            )
        )

    for case_key, slack_value, top_k, scale, threshold in [
        ("builtin_pjm5", 1.015, 4, 1.050, 1.700),
        ("uploaded_pjm5", 1.025, 3, 0.970, 1.200),
    ]:
        add(
            _build_voltage_then_line_conversation(
                CASES[case_key],
                scenario_number,
                slack_value=slack_value,
                top_k=top_k,
                scale=scale,
                threshold=threshold,
            )
        )

    return scenarios


def validate_scenario(scenario: ConversationScenario) -> Dict[str, Any]:
    import andes

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfagent-generalized")

    turn_results: List[Dict[str, Any]] = []
    all_passed = True

    with tempfile.TemporaryDirectory(prefix=f"pfagent-generalized-{scenario.scenario_id}-", dir="/tmp") as tmpdir:
        root = Path(tmpdir)
        for turn in scenario.turns:
            runtime_dir = root / f"turn_{turn.turn_id:02d}"
            runtime_dir.mkdir(parents=True, exist_ok=True)

            for filename, source_case in scenario.uploaded_files.items():
                shutil.copyfile(andes.get_case(source_case), runtime_dir / filename)

            script_path = runtime_dir / "scenario.py"
            script_path.write_text(turn.assistant, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, script_path.name],
                cwd=runtime_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )

            actual_stdout = _normalize_stdout(result.stdout)
            actual_stderr = _normalize_stdout(result.stderr)
            stdout_match = actual_stdout == _normalize_stdout(turn.expected_stdout)
            artifacts_present = all((runtime_dir / artifact).exists() for artifact in turn.artifact_files)
            passed = result.returncode == 0 and stdout_match and artifacts_present
            all_passed = all_passed and passed

            turn_results.append(
                {
                    "turn_id": turn.turn_id,
                    "passed": passed,
                    "returncode": result.returncode,
                    "stdout_match": stdout_match,
                    "artifacts_present": artifacts_present,
                    "expected_stdout": turn.expected_stdout,
                    "actual_stdout": actual_stdout,
                    "stderr": actual_stderr,
                    "artifact_files": turn.artifact_files,
                }
            )

    return {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "case_key": scenario.case_key,
        "passed": all_passed,
        "turn_results": turn_results,
    }


def write_examples(scenarios: Sequence[ConversationScenario], output_path: Path) -> None:
    payload = {
        "examples": [
            {
                "id": scenario.scenario_id,
                "family": scenario.family,
                "case_key": scenario.case_key,
                "messages": scenario.to_messages(),
                "validation": {
                    "source_case_path": scenario.source_case_path,
                    "uploaded_filename": scenario.uploaded_filename,
                    "turns": [
                        {
                            "turn_id": turn.turn_id,
                            "expected_stdout": turn.expected_stdout,
                            "artifact_files": turn.artifact_files,
                        }
                        for turn in scenario.turns
                    ],
                },
            }
            for scenario in scenarios
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_report(results: Sequence[Dict[str, Any]], output_path: Path) -> None:
    all_turns = [turn for result in results for turn in result["turn_results"]]
    summary = {
        "total_scenarios": len(results),
        "passed_scenarios": sum(1 for item in results if item["passed"]),
        "failed_scenarios": sum(1 for item in results if not item["passed"]),
        "total_turns": len(all_turns),
        "passed_turns": sum(1 for item in all_turns if item["passed"]),
        "failed_turns": sum(1 for item in all_turns if not item["passed"]),
        "results": list(results),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate generalized multi-turn verified fine-tuning examples.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_JSON),
        help="Path for the generalized verified examples JSON file.",
    )
    parser.add_argument(
        "--report-output",
        default=str(REPORT_JSON),
        help="Path for the validation report JSON file.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write only passing conversations even if some scenarios fail validation.",
    )
    args = parser.parse_args()

    scenarios = build_scenarios()
    results = [validate_scenario(scenario) for scenario in scenarios]
    passed_ids = {item["scenario_id"] for item in results if item["passed"]}
    passing_scenarios = [scenario for scenario in scenarios if scenario.scenario_id in passed_ids]

    write_examples(passing_scenarios, Path(args.output))
    write_report(results, Path(args.report_output))

    failed = [item for item in results if not item["passed"]]
    print(f"Generalized verified examples written to: {args.output}")
    print(f"Validation report written to: {args.report_output}")
    print(f"Validation summary: {len(passing_scenarios)}/{len(scenarios)} conversations passed")

    for item in failed:
        print(f"- FAIL: {item['scenario_id']}")
        for turn in item["turn_results"]:
            if turn["passed"]:
                continue
            print(
                "  turn=%s returncode=%s stdout_match=%s artifacts_present=%s"
                % (turn["turn_id"], turn["returncode"], turn["stdout_match"], turn["artifacts_present"])
            )

    if failed and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
