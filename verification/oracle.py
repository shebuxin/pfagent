from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import andes
import numpy as np


_ORACLE_PYCODE_PATH = Path("/tmp/pfagent-andes-pycode")


def _round_float(value: Any) -> float:
    return round(float(value), 6)


def _load_case(scenario: Dict[str, Any]):
    _ORACLE_PYCODE_PATH.mkdir(parents=True, exist_ok=True)
    return andes.load(
        andes.get_case(scenario["source_case_path"]),
        setup=False,
        no_output=True,
        default_config=True,
        log=False,
        pycode_path=str(_ORACLE_PYCODE_PATH),
    )


def _apply_operations(ssa, operations: List[Dict[str, Any]]) -> None:
    add_ops = [op for op in operations if op["type"] == "add_pq"]
    post_setup_ops = [op for op in operations if op["type"] not in {"add_pq", "n1_screening"}]

    for op in add_ops:
        ssa.add(
            "PQ",
            param_dict={
                "bus": op["bus"],
                "idx": op["idx"],
                "p0": op["p0"],
                "q0": op["q0"],
            },
        )

    ssa.setup()

    for op in post_setup_ops:
        if op["type"] == "scale_all_pq":
            factor = op["factor"]
            ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=factor * ssa.PQ.p0.v)
            ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=factor * ssa.PQ.q0.v)
        elif op["type"] == "set_slack_v0":
            ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[op["value"]])
        elif op["type"] == "set_first_pv_v0":
            ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[op["value"]])
        elif op["type"] == "scale_pq_at_bus":
            pq_bus = np.asarray(ssa.PQ.bus.v)
            mask = pq_bus == int(op["bus"])
            target_idx = np.asarray(ssa.PQ.idx.v, dtype=object)[mask].tolist()
            if not target_idx:
                raise KeyError(f"Missing PQ device at bus {op['bus']}")
            target_p0 = np.asarray(ssa.PQ.p0.v, dtype=float)[mask]
            target_q0 = np.asarray(ssa.PQ.q0.v, dtype=float)[mask]
            factor = float(op["factor"])
            ssa.PQ.set(src="p0", idx=target_idx, attr="v", value=factor * target_p0)
            ssa.PQ.set(src="q0", idx=target_idx, attr="v", value=factor * target_q0)
        elif op["type"] == "set_pv_bus_v0":
            pv_bus = np.asarray(ssa.PV.bus.v)
            mask = pv_bus == int(op["bus"])
            target_idx = np.asarray(ssa.PV.idx.v, dtype=object)[mask].tolist()
            if not target_idx:
                raise KeyError(f"Missing PV device at bus {op['bus']}")
            ssa.PV.set(
                src="v0",
                idx=target_idx,
                attr="v",
                value=np.full(len(target_idx), float(op["value"]), dtype=float),
            )
        elif op["type"] == "line_outage_by_pair":
            ssa.Line.set(src="u", idx=op["line_id"], attr="v", value=0)
        else:
            raise ValueError(f"Unsupported operation type: {op['type']}")

    ssa.PFlow.run()


def _top_indices(values: np.ndarray, top_k: int, highest: bool) -> np.ndarray:
    order = np.argsort(values)
    if highest:
        order = order[::-1]
    return order[:top_k]


def _bus_arrays(ssa):
    return np.asarray(ssa.Bus.idx.v), np.asarray(ssa.Bus.v.v, dtype=float)


def _line_arrays(ssa):
    return np.asarray(ssa.Line.idx.v), np.abs(np.asarray(ssa.Line.a1.e, dtype=float))


def _contingency_status_label(status: Dict[str, Any]) -> str:
    if status["no_slack_islands"] > 0:
        return "no_slack_island"
    if status["multi_slack_islands"] > 0:
        return "multi_slack_island"
    if not status["converged"]:
        return "not_converged"
    if status["island_count"] > 1 or status["islanded_bus_count"] > 0:
        return "converged_with_islanding"
    return "converged"


def _collect_pflow_status(ssa) -> Dict[str, Any]:
    bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    island_sets = list(getattr(ssa.Bus, "island_sets", []) or [])
    nosw_island = list(getattr(ssa.Bus, "nosw_island", []) or [])
    msw_island = list(getattr(ssa.Bus, "msw_island", []) or [])
    min_pos = int(np.argmin(bus_v)) if bus_v.size else 0
    mismatch_series = getattr(ssa.PFlow, "mis", None)
    last_mismatch = None
    if mismatch_series is not None and len(mismatch_series):
        last_mismatch = _round_float(mismatch_series[-1])

    status = {
        "converged": bool(getattr(ssa.PFlow, "converged", False)),
        "exit_code": int(getattr(ssa, "exit_code", 1)),
        "island_count": int(len(island_sets)),
        "no_slack_islands": int(len(nosw_island)),
        "multi_slack_islands": int(len(msw_island)),
        "islanded_bus_count": int(getattr(ssa.Bus, "n_islanded_buses", 0) or 0),
        "all_finite_voltages": bool(np.all(np.isfinite(bus_v))) if bus_v.size else False,
        "min_bus": int(bus_ids[min_pos]) if bus_v.size else None,
        "min_voltage": _round_float(bus_v[min_pos]) if bus_v.size else None,
        "last_mismatch": last_mismatch,
    }
    status["outage_status"] = _contingency_status_label(status)
    return status


def _failure_aware_priority(record: Dict[str, Any]) -> tuple:
    status = str(record["outage_status"])
    severity = {
        "no_slack_island": 4,
        "multi_slack_island": 3,
        "not_converged": 2,
        "converged_with_islanding": 1,
        "converged": 0,
    }.get(status, 0)
    min_voltage = float(record["min_voltage"]) if record["min_voltage"] is not None else float("inf")
    last_mismatch = float(record["last_mismatch"]) if record["last_mismatch"] is not None else -1.0
    return (
        severity,
        int(record["no_slack_islands"]),
        int(record["multi_slack_islands"]),
        int(record["island_count"]),
        int(record["islanded_bus_count"]),
        -min_voltage,
        last_mismatch,
    )


def _find_pq_at_bus(ssa, bus: int):
    pq_buses = np.asarray(ssa.PQ.bus.v)
    mask = pq_buses == int(bus)
    if not np.any(mask):
        raise KeyError(f"Missing PQ device at bus {bus}")
    idx_values = np.asarray(ssa.PQ.idx.v, dtype=object)[mask]
    p0_values = np.asarray(ssa.PQ.p0.v, dtype=float)[mask]
    q0_values = np.asarray(ssa.PQ.q0.v, dtype=float)[mask]
    return str(idx_values[0]), _round_float(p0_values[0]), _round_float(q0_values[0])


def _find_pv_at_bus(ssa, bus: int):
    pv_buses = np.asarray(ssa.PV.bus.v)
    mask = pv_buses == int(bus)
    if not np.any(mask):
        raise KeyError(f"Missing PV device at bus {bus}")
    idx_values = np.asarray(ssa.PV.idx.v, dtype=object)[mask]
    return str(idx_values[0]), _round_float(np.asarray(ssa.PV.v0.v, dtype=float)[mask][0])


def _latest_op(operations: List[Dict[str, Any]], op_type: str) -> Dict[str, Any]:
    for op in reversed(operations):
        if op["type"] == op_type:
            return op
    raise KeyError(f"Missing operation of type {op_type}")


def _report(scenario: Dict[str, Any], ssa, turn: Dict[str, Any]) -> Dict[str, Any]:
    bus_ids, bus_v = _bus_arrays(ssa)
    slack_bus = int(ssa.Slack.bus.v[0])
    slack_voltage = _round_float(bus_v[np.where(bus_ids == slack_bus)[0][0]])
    ops = turn["cumulative_operations"]
    report_kind = turn["report_kind"]
    params = turn.get("task_params", {})

    if report_kind == "baseline_high_rank_report":
        idx = _top_indices(bus_v, int(params["top_k"]), highest=True)
        return {
            "slack_bus": slack_bus,
            "slack_voltage": slack_voltage,
            "selected_bus_ids": [int(bus_ids[i]) for i in idx],
            "selected_voltages": [_round_float(bus_v[i]) for i in idx],
        }

    if report_kind == "add_load_threshold_report":
        op = _latest_op(ops, "add_pq")
        threshold = float(params["threshold"])
        mask = bus_v < threshold
        below_ids = [int(x) for x in bus_ids[mask]]
        min_i = int(np.argmin(bus_v))
        return {
            "added_load_idx": op["idx"],
            "added_load_bus": int(op["bus"]),
            "threshold": _round_float(threshold),
            "selected_bus_ids": below_ids,
            "selected_count": len(below_ids),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }

    if report_kind == "scaled_plot_report":
        op = _latest_op(ops, "scale_all_pq")
        min_i = int(np.argmin(bus_v))
        max_i = int(np.argmax(bus_v))
        return {
            "scale_factor": _round_float(op["factor"]),
            "max_bus": int(bus_ids[max_i]),
            "max_voltage": _round_float(bus_v[max_i]),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
            "plot_file": turn["plot_filename"],
        }

    if report_kind == "baseline_threshold_low_rank_report":
        threshold = float(params["threshold"])
        mask = bus_v > threshold
        low_idx = _top_indices(bus_v, 2, highest=False)
        return {
            "threshold": _round_float(threshold),
            "selected_bus_ids": [int(x) for x in bus_ids[mask]],
            "selected_count": int(np.sum(mask)),
            "lowest_bus_ids": [int(bus_ids[i]) for i in low_idx],
            "lowest_voltages": [_round_float(bus_v[i]) for i in low_idx],
        }

    if report_kind == "slack_adjust_report":
        op = _latest_op(ops, "set_slack_v0")
        threshold = float(params["threshold"])
        count = int(np.sum(bus_v < threshold))
        return {
            "slack_bus": slack_bus,
            "slack_setpoint": _round_float(op["value"]),
            "slack_voltage": slack_voltage,
            "selected_count": count,
        }

    if report_kind == "extremes_report":
        max_i = int(np.argmax(bus_v))
        min_i = int(np.argmin(bus_v))
        result = {
            "max_bus": int(bus_ids[max_i]),
            "max_voltage": _round_float(bus_v[max_i]),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }
        if any(op["type"] == "add_pq" for op in ops):
            result["added_load_idx"] = _latest_op(ops, "add_pq")["idx"]
        if any(op["type"] == "add_pq" for op in ops) or turn["result_keys"][-1] == "total_pq_count":
            result["total_pq_count"] = int(len(ssa.PQ.idx.v))
        return result

    if report_kind == "pv_adjust_report":
        op = _latest_op(ops, "set_first_pv_v0")
        threshold = float(params["threshold"])
        pv_bus = int(ssa.PV.bus.v[0])
        pv_i = int(np.where(bus_ids == pv_bus)[0][0])
        return {
            "pv_bus": pv_bus,
            "pv_setpoint": _round_float(op["value"]),
            "pv_voltage": _round_float(bus_v[pv_i]),
            "selected_count": int(np.sum(bus_v > threshold)),
        }

    if report_kind == "scaled_bar_plot_report":
        op = _latest_op(ops, "scale_all_pq")
        min_i = int(np.argmin(bus_v))
        max_i = int(np.argmax(bus_v))
        return {
            "scale_factor": _round_float(op["factor"]),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
            "max_bus": int(bus_ids[max_i]),
            "max_voltage": _round_float(bus_v[max_i]),
            "plot_file": turn["plot_filename"],
        }

    if report_kind == "baseline_low_rank_report":
        idx = _top_indices(bus_v, int(params["top_k"]), highest=False)
        return {
            "selected_bus_ids": [int(bus_ids[i]) for i in idx],
            "selected_voltages": [_round_float(bus_v[i]) for i in idx],
        }

    if report_kind == "add_load_slack_threshold_report":
        op = _latest_op(ops, "add_pq")
        threshold = float(params["threshold"])
        mask = bus_v < threshold
        return {
            "added_load_idx": op["idx"],
            "slack_bus": slack_bus,
            "slack_voltage": slack_voltage,
            "threshold": _round_float(threshold),
            "selected_bus_ids": [int(x) for x in bus_ids[mask]],
            "selected_count": int(np.sum(mask)),
        }

    if report_kind == "slack_plot_low_rank_report":
        op = _latest_op(ops, "set_slack_v0")
        idx = _top_indices(bus_v, int(params["top_k"]), highest=False)
        return {
            "slack_setpoint": _round_float(op["value"]),
            "slack_voltage": slack_voltage,
            "selected_bus_ids": [int(bus_ids[i]) for i in idx],
            "selected_voltages": [_round_float(bus_v[i]) for i in idx],
            "plot_file": turn["plot_filename"],
        }

    if report_kind == "line_topk_report":
        line_ids, abs_a1 = _line_arrays(ssa)
        idx = _top_indices(abs_a1, int(params["top_k"]), highest=True)
        return {
            "selected_line_ids": [str(line_ids[i]) for i in idx],
            "selected_line_metrics": [_round_float(abs_a1[i]) for i in idx],
        }

    if report_kind == "scaled_line_threshold_report":
        line_ids, abs_a1 = _line_arrays(ssa)
        op = _latest_op(ops, "scale_all_pq")
        threshold = float(params["angle_threshold"])
        mask = abs_a1 > threshold
        return {
            "scale_factor": _round_float(op["factor"]),
            "angle_threshold": _round_float(threshold),
            "selected_line_ids": [str(x) for x in line_ids[mask]],
            "selected_count": int(np.sum(mask)),
        }

    if report_kind == "add_load_voltage_plot_report":
        op = _latest_op(ops, "add_pq")
        min_i = int(np.argmin(bus_v))
        max_i = int(np.argmax(bus_v))
        return {
            "added_load_idx": op["idx"],
            "max_bus": int(bus_ids[max_i]),
            "max_voltage": _round_float(bus_v[max_i]),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
            "plot_file": turn["plot_filename"],
        }

    if report_kind == "baseline_slack_extremes_report":
        max_i = int(np.argmax(bus_v))
        min_i = int(np.argmin(bus_v))
        return {
            "slack_bus": slack_bus,
            "slack_voltage": slack_voltage,
            "max_bus": int(bus_ids[max_i]),
            "max_voltage": _round_float(bus_v[max_i]),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }

    if report_kind == "slack_line_topk_report":
        line_ids, abs_a1 = _line_arrays(ssa)
        op = _latest_op(ops, "set_slack_v0")
        idx = _top_indices(abs_a1, int(params["top_k"]), highest=True)
        return {
            "slack_setpoint": _round_float(op["value"]),
            "slack_voltage": slack_voltage,
            "selected_line_ids": [str(line_ids[i]) for i in idx],
            "selected_line_metrics": [_round_float(abs_a1[i]) for i in idx],
        }

    if report_kind == "slack_scaled_line_threshold_report":
        line_ids, abs_a1 = _line_arrays(ssa)
        slack_op = _latest_op(ops, "set_slack_v0")
        scale_op = _latest_op(ops, "scale_all_pq")
        threshold = float(params["angle_threshold"])
        mask = abs_a1 > threshold
        return {
            "slack_setpoint": _round_float(slack_op["value"]),
            "scale_factor": _round_float(scale_op["factor"]),
            "angle_threshold": _round_float(threshold),
            "selected_line_ids": [str(x) for x in line_ids[mask]],
            "selected_count": int(np.sum(mask)),
        }

    if report_kind == "pq_bus_inspection_report":
        target_bus = int(params["target_bus"])
        target_idx, target_p0, target_q0 = _find_pq_at_bus(ssa, target_bus)
        return {
            "target_pq_bus": target_bus,
            "target_pq_idx": target_idx,
            "target_p0": target_p0,
            "target_q0": target_q0,
            "slack_bus": slack_bus,
            "slack_voltage": slack_voltage,
        }

    if report_kind == "pq_bus_scale_report":
        op = _latest_op(ops, "scale_pq_at_bus")
        target_idx, target_p0, target_q0 = _find_pq_at_bus(ssa, int(op["bus"]))
        min_i = int(np.argmin(bus_v))
        return {
            "target_pq_bus": int(op["bus"]),
            "target_pq_idx": target_idx,
            "scale_factor": _round_float(op["factor"]),
            "target_p0": target_p0,
            "target_q0": target_q0,
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }

    if report_kind == "pq_bus_scale_threshold_report":
        op = _latest_op(ops, "scale_pq_at_bus")
        threshold = float(params["threshold"])
        target_idx, target_p0, target_q0 = _find_pq_at_bus(ssa, int(op["bus"]))
        mask = bus_v < threshold
        return {
            "target_pq_bus": int(op["bus"]),
            "target_pq_idx": target_idx,
            "scale_factor": _round_float(op["factor"]),
            "target_p0": target_p0,
            "target_q0": target_q0,
            "threshold": _round_float(threshold),
            "selected_bus_ids": [int(x) for x in bus_ids[mask]],
            "selected_count": int(np.sum(mask)),
        }

    if report_kind == "n1_screening_report":
        scale_op = _latest_op(ops, "scale_pq_at_bus")
        base_ops = [op for op in ops if op["type"] != "n1_screening"]
        candidate_lines = params["candidate_lines"]
        screened_ids = [str(item["line_id"]) for item in candidate_lines]
        worst = None
        for candidate in candidate_lines:
            contingency_ssa = _load_case(scenario)
            _apply_operations(contingency_ssa, base_ops)
            contingency_ssa.Line.set(src="u", idx=candidate["line_id"], attr="v", value=0)
            contingency_ssa.PFlow.run()
            status = _collect_pflow_status(contingency_ssa)
            record = {
                "line_id": str(candidate["line_id"]),
                "bus_pair": [int(candidate["bus1"]), int(candidate["bus2"])],
                "min_bus": status["min_bus"],
                "min_voltage": status["min_voltage"],
            }
            record_min_voltage = float(record["min_voltage"]) if record["min_voltage"] is not None else float("inf")
            worst_min_voltage = float(worst["min_voltage"]) if worst and worst["min_voltage"] is not None else float("inf")
            if worst is None or record_min_voltage < worst_min_voltage:
                worst = record
        return {
            "scale_factor": _round_float(scale_op["factor"]),
            "candidate_line_ids": screened_ids,
            "worst_line_id": worst["line_id"],
            "worst_line_bus_pair": worst["bus_pair"],
            "worst_min_bus": worst["min_bus"],
            "worst_min_voltage": worst["min_voltage"],
        }

    if report_kind == "n1_failure_aware_screening_report":
        scale_op = _latest_op(ops, "scale_pq_at_bus")
        base_ops = [op for op in ops if op["type"] != "n1_screening"]
        candidate_lines = params["candidate_lines"]
        screened_ids = [str(item["line_id"]) for item in candidate_lines]
        worst = None
        for candidate in candidate_lines:
            contingency_ssa = _load_case(scenario)
            _apply_operations(contingency_ssa, base_ops)
            contingency_ssa.Line.set(src="u", idx=candidate["line_id"], attr="v", value=0)
            contingency_ssa.PFlow.run()
            status = _collect_pflow_status(contingency_ssa)
            record = {
                "line_id": str(candidate["line_id"]),
                "bus_pair": [int(candidate["bus1"]), int(candidate["bus2"])],
                "outage_status": status["outage_status"],
                "exit_code": status["exit_code"],
                "island_count": status["island_count"],
                "no_slack_islands": status["no_slack_islands"],
                "islanded_bus_count": status["islanded_bus_count"],
                "min_bus": status["min_bus"],
                "min_voltage": status["min_voltage"],
                "last_mismatch": status["last_mismatch"],
                "multi_slack_islands": status["multi_slack_islands"],
            }
            if worst is None or _failure_aware_priority(record) > _failure_aware_priority(worst):
                worst = record
        return {
            "scale_factor": _round_float(scale_op["factor"]),
            "candidate_line_ids": screened_ids,
            "worst_line_id": worst["line_id"],
            "worst_line_bus_pair": worst["bus_pair"],
            "worst_outage_status": worst["outage_status"],
            "worst_exit_code": int(worst["exit_code"]),
            "worst_island_count": int(worst["island_count"]),
            "worst_no_slack_islands": int(worst["no_slack_islands"]),
            "worst_islanded_bus_count": int(worst["islanded_bus_count"]),
            "worst_min_bus": worst["min_bus"],
            "worst_min_voltage": worst["min_voltage"],
        }

    if report_kind == "pv_bus_inspection_report":
        target_bus = int(params["target_bus"])
        pv_idx, pv_setpoint = _find_pv_at_bus(ssa, target_bus)
        pv_i = int(np.where(bus_ids == target_bus)[0][0])
        return {
            "pv_bus": target_bus,
            "pv_idx": pv_idx,
            "pv_setpoint": pv_setpoint,
            "pv_voltage": _round_float(bus_v[pv_i]),
        }

    if report_kind == "pv_bus_adjust_threshold_report":
        op = _latest_op(ops, "set_pv_bus_v0")
        pv_idx, pv_setpoint = _find_pv_at_bus(ssa, int(op["bus"]))
        threshold = float(params["threshold"])
        pv_i = int(np.where(bus_ids == int(op["bus"]))[0][0])
        return {
            "pv_bus": int(op["bus"]),
            "pv_idx": pv_idx,
            "pv_setpoint": pv_setpoint,
            "pv_voltage": _round_float(bus_v[pv_i]),
            "threshold": _round_float(threshold),
            "selected_count": int(np.sum(bus_v > threshold)),
        }

    if report_kind == "pv_line_outage_report":
        pv_op = _latest_op(ops, "set_pv_bus_v0")
        line_op = _latest_op(ops, "line_outage_by_pair")
        min_i = int(np.argmin(bus_v))
        return {
            "pv_setpoint": _round_float(pv_op["value"]),
            "opened_line_id": str(line_op["line_id"]),
            "opened_line_bus_pair": [int(line_op["bus1"]), int(line_op["bus2"])],
            "slack_bus": slack_bus,
            "slack_voltage": slack_voltage,
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }

    if report_kind == "pq_line_outage_threshold_report":
        pq_op = _latest_op(ops, "scale_pq_at_bus")
        line_op = _latest_op(ops, "line_outage_by_pair")
        threshold = float(params["threshold"])
        min_i = int(np.argmin(bus_v))
        mask = bus_v < threshold
        return {
            "scale_factor": _round_float(pq_op["factor"]),
            "opened_line_id": str(line_op["line_id"]),
            "opened_line_bus_pair": [int(line_op["bus1"]), int(line_op["bus2"])],
            "threshold": _round_float(threshold),
            "selected_bus_ids": [int(x) for x in bus_ids[mask]],
            "selected_count": int(np.sum(mask)),
            "min_bus": int(bus_ids[min_i]),
            "min_voltage": _round_float(bus_v[min_i]),
        }

    raise ValueError(f"Unsupported report kind: {report_kind}")


def compute_oracle_turn_result(scenario: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
    ssa = _load_case(scenario)
    _apply_operations(ssa, turn["cumulative_operations"])
    return _report(scenario, ssa, turn)


def compute_oracle_for_scenario(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [compute_oracle_turn_result(scenario, turn) for turn in scenario["turns"]]
