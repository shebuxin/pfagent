"""Structured ANDES script builders.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. One
builder per report-kind family plus the shared case-load-lines helper.
Each builder returns a full runnable Python script as a string. These
are the biggest templates in the codebase (~860 lines total).

All dependencies are already-extracted state helpers from
``src.andes_code.structured.state``.
"""

from __future__ import annotations

from typing import List

from src.andes_code.structured.state import (
    StructuredAndesState,
    extract_first_float,
    extract_high_voltage_threshold,
    extract_low_voltage_threshold,
    extract_plot_filename,
    extract_result_json_keys,
    extract_top_k_from_prompt,
)


def _build_structured_case_load_lines(state: StructuredAndesState) -> List[str]:
    if state.case_source == "uploaded":
        return [
            "script_dir = os.getcwd()",
            f'case = os.path.join(script_dir, "{state.case_reference}")',
            'ssa = andes.load(case, setup=False, no_output=True, log=False)',
        ]

    return [
        "ssa = andes.load(",
        f'    andes.get_case("{state.case_reference}"),',
        "    setup=False,",
        "    no_output=True,",
        "    log=False,",
        ")",
    ]


def build_structured_targeted_pq_script(
    report_kind: str,
    user_context: str,
    state: StructuredAndesState,
) -> str:
    target_bus = state.target_pq_bus
    if target_bus is None:
        raise ValueError("Targeted PQ structured codegen requires a target bus.")

    threshold = extract_low_voltage_threshold(user_context)
    required_dependencies = ["andes", "numpy"]
    imports = ["import json", "import andes", "import numpy as np"]
    if state.case_source == "uploaded":
        imports.append("import os")

    lines: List[str] = [
        f"# required_dependencies: {','.join(required_dependencies)}",
        *imports,
        "",
        *_build_structured_case_load_lines(state),
        "",
        "ssa.setup()",
        "",
        f"target_pq_bus = {target_bus}",
        "pq_bus_array = np.asarray(ssa.PQ.bus.v, dtype=int)",
        "pq_positions = np.where(pq_bus_array == target_pq_bus)[0]",
        'if pq_positions.size == 0:',
        '    raise ValueError(f\"No PQ device found at bus {target_pq_bus}\")',
        "pq_pos = int(pq_positions[0])",
        "target_pq_idx_value = ssa.PQ.idx.v[pq_pos]",
        "target_pq_idx = str(target_pq_idx_value)",
    ]

    if report_kind in {"pq_bus_scale_report", "pq_bus_scale_threshold_report", "pq_line_outage_threshold_report"}:
        scale_factor = state.target_pq_scale_factor
        lines.extend(
            [
                f"scale_factor = {scale_factor}",
                "target_p0_before = float(ssa.PQ.p0.v[pq_pos])",
                "target_q0_before = float(ssa.PQ.q0.v[pq_pos])",
                'ssa.PQ.set(src="p0", idx=[target_pq_idx_value], attr="v", value=[target_p0_before * scale_factor])',
                'ssa.PQ.set(src="q0", idx=[target_pq_idx_value], attr="v", value=[target_q0_before * scale_factor])',
            ]
        )

    if report_kind == "pq_line_outage_threshold_report":
        bus1, bus2 = state.opened_line_pair or (None, None)
        lines.extend(
            [
                f"opened_bus1 = {bus1}",
                f"opened_bus2 = {bus2}",
                "line_bus1 = np.asarray(ssa.Line.bus1.v, dtype=int)",
                "line_bus2 = np.asarray(ssa.Line.bus2.v, dtype=int)",
                "line_positions = np.where(((line_bus1 == opened_bus1) & (line_bus2 == opened_bus2)) | ((line_bus1 == opened_bus2) & (line_bus2 == opened_bus1)))[0]",
                "if line_positions.size == 0:",
                '    raise ValueError(f\"No line found for bus pair {opened_bus1}-{opened_bus2}\")',
                "line_pos = int(line_positions[0])",
                "opened_line_id_value = ssa.Line.idx.v[line_pos]",
                "opened_line_id = str(opened_line_id_value)",
                'ssa.Line.set(src="u", idx=opened_line_id_value, attr="v", value=0)',
            ]
        )

    lines.extend(
        [
            "",
            (
                "run_result = ssa.PFlow.run()\n"
                "try:\n"
                "    converged = bool(ssa.PFlow.converged)\n"
                "except Exception:\n"
                "    converged = bool(run_result)"
            )
            if report_kind == "pq_line_outage_threshold_report"
            else "ssa.PFlow.run()"
            ,
            "",
            "bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)",
            "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
            "target_p0 = round(float(ssa.PQ.p0.v[pq_pos]), 6)",
            "target_q0 = round(float(ssa.PQ.q0.v[pq_pos]), 6)",
            "result_json = {}",
        ]
    )

    if report_kind == "pq_line_outage_threshold_report":
        lines[-1:-1] = [
            "island_sets_raw = ssa.Bus.island_sets if hasattr(ssa.Bus, 'island_sets') else []",
            "island_sets = list(island_sets_raw or [])",
            "no_slack_raw = ssa.Bus.nosw_island if hasattr(ssa.Bus, 'nosw_island') else []",
            "no_slack_islands = int(len(no_slack_raw or []))",
            "islanded_bus_count = int(ssa.Bus.n_islanded_buses if hasattr(ssa.Bus, 'n_islanded_buses') else 0)",
            "exit_code = int(getattr(ssa, 'exit_code', 1))",
            "if not converged or no_slack_islands > 0:",
            "    raise RuntimeError(f'Post-outage power flow did not converge cleanly (exit_code={exit_code}, islands={len(island_sets)}, no_slack_islands={no_slack_islands}, islanded_bus_count={islanded_bus_count}).')",
        ]

    if report_kind == "pq_bus_inspection_report":
        lines.extend(
            [
                "slack_bus = int(ssa.Slack.bus.v[0])",
                "slack_pos = int(np.where(bus_ids == slack_bus)[0][0])",
                "slack_voltage = round(float(bus_v[slack_pos]), 6)",
                'result_json["target_pq_bus"] = target_pq_bus',
                'result_json["target_pq_idx"] = target_pq_idx',
                'result_json["target_p0"] = target_p0',
                'result_json["target_q0"] = target_q0',
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_voltage"] = slack_voltage',
            ]
        )
    elif report_kind == "pq_bus_scale_threshold_report":
        threshold = threshold if threshold is not None else 0.95
        lines.extend(
            [
                f"threshold = {threshold}",
                "below_mask = bus_v < threshold",
                'result_json["target_pq_bus"] = target_pq_bus',
                'result_json["target_pq_idx"] = target_pq_idx',
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["target_p0"] = target_p0',
                'result_json["target_q0"] = target_q0',
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]',
                'result_json["selected_count"] = int(np.sum(below_mask))',
            ]
        )
    elif report_kind == "pq_line_outage_threshold_report":
        threshold = threshold if threshold is not None else 0.95
        lines.extend(
            [
                f"threshold = {threshold}",
                "below_mask = bus_v < threshold",
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["opened_line_id"] = opened_line_id',
                'result_json["opened_line_bus_pair"] = [opened_bus1, opened_bus2]',
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]',
                'result_json["selected_count"] = int(np.sum(below_mask))',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )
    else:
        lines.extend(
            [
                'result_json["target_pq_bus"] = target_pq_bus',
                'result_json["target_pq_idx"] = target_pq_idx',
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["target_p0"] = target_p0',
                'result_json["target_q0"] = target_q0',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )

    lines.extend(["", 'print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))'])
    return "\n".join(lines)


def build_structured_targeted_pv_script(
    report_kind: str,
    user_context: str,
    state: StructuredAndesState,
) -> str:
    target_bus = state.target_pv_bus
    if target_bus is None:
        raise ValueError("Targeted PV structured codegen requires a target bus.")

    threshold = extract_high_voltage_threshold(user_context)
    required_dependencies = ["andes", "numpy"]
    imports = ["import json", "import andes", "import numpy as np"]
    if state.case_source == "uploaded":
        imports.append("import os")

    lines: List[str] = [
        f"# required_dependencies: {','.join(required_dependencies)}",
        *imports,
        "",
        *_build_structured_case_load_lines(state),
        "",
        "ssa.setup()",
        "",
        f"target_pv_bus = {target_bus}",
        "pv_bus_array = np.asarray(ssa.PV.bus.v, dtype=int)",
        "pv_positions = np.where(pv_bus_array == target_pv_bus)[0]",
        'if pv_positions.size == 0:',
        '    raise ValueError(f\"No PV device found at bus {target_pv_bus}\")',
        "pv_pos = int(pv_positions[0])",
        "target_pv_idx_value = ssa.PV.idx.v[pv_pos]",
        "target_pv_idx = str(target_pv_idx_value)",
    ]

    if report_kind in {"pv_bus_adjust_threshold_report", "pv_line_outage_report"}:
        pv_setpoint = state.target_pv_setpoint
        lines.extend(
            [
                f"pv_setpoint = {pv_setpoint}",
                'ssa.PV.set(src="v0", idx=[target_pv_idx_value], attr="v", value=[pv_setpoint])',
            ]
        )

    if report_kind == "pv_line_outage_report":
        bus1, bus2 = state.opened_line_pair or (None, None)
        lines.extend(
            [
                f"opened_bus1 = {bus1}",
                f"opened_bus2 = {bus2}",
                "line_bus1 = np.asarray(ssa.Line.bus1.v, dtype=int)",
                "line_bus2 = np.asarray(ssa.Line.bus2.v, dtype=int)",
                "line_positions = np.where(((line_bus1 == opened_bus1) & (line_bus2 == opened_bus2)) | ((line_bus1 == opened_bus2) & (line_bus2 == opened_bus1)))[0]",
                "if line_positions.size == 0:",
                '    raise ValueError(f\"No line found for bus pair {opened_bus1}-{opened_bus2}\")',
                "line_pos = int(line_positions[0])",
                "opened_line_id_value = ssa.Line.idx.v[line_pos]",
                "opened_line_id = str(opened_line_id_value)",
                'ssa.Line.set(src="u", idx=opened_line_id_value, attr="v", value=0)',
            ]
        )

    lines.extend(
        [
            "",
            (
                "run_result = ssa.PFlow.run()\n"
                "try:\n"
                "    converged = bool(ssa.PFlow.converged)\n"
                "except Exception:\n"
                "    converged = bool(run_result)"
            )
            if report_kind == "pv_line_outage_report"
            else "ssa.PFlow.run()"
            ,
            "",
            "bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)",
            "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
            "pv_voltage_pos = int(np.where(bus_ids == target_pv_bus)[0][0])",
            "pv_voltage = round(float(bus_v[pv_voltage_pos]), 6)",
            "current_pv_setpoint = round(float(ssa.PV.v0.v[pv_pos]), 6)",
            "result_json = {}",
        ]
    )

    if report_kind == "pv_line_outage_report":
        lines[-1:-1] = [
            "island_sets_raw = ssa.Bus.island_sets if hasattr(ssa.Bus, 'island_sets') else []",
            "island_sets = list(island_sets_raw or [])",
            "no_slack_raw = ssa.Bus.nosw_island if hasattr(ssa.Bus, 'nosw_island') else []",
            "no_slack_islands = int(len(no_slack_raw or []))",
            "islanded_bus_count = int(ssa.Bus.n_islanded_buses if hasattr(ssa.Bus, 'n_islanded_buses') else 0)",
            "exit_code = int(getattr(ssa, 'exit_code', 1))",
            "if not converged or no_slack_islands > 0:",
            "    raise RuntimeError(f'Post-outage power flow did not converge cleanly (exit_code={exit_code}, islands={len(island_sets)}, no_slack_islands={no_slack_islands}, islanded_bus_count={islanded_bus_count}).')",
        ]

    if report_kind == "pv_bus_inspection_report":
        lines.extend(
            [
                'result_json["pv_bus"] = target_pv_bus',
                'result_json["pv_idx"] = target_pv_idx',
                'result_json["pv_setpoint"] = current_pv_setpoint',
                'result_json["pv_voltage"] = pv_voltage',
            ]
        )
    elif report_kind == "pv_bus_adjust_threshold_report":
        threshold = threshold if threshold is not None else 1.0
        lines.extend(
            [
                f"threshold = {threshold}",
                'result_json["pv_bus"] = target_pv_bus',
                'result_json["pv_idx"] = target_pv_idx',
                'result_json["pv_setpoint"] = current_pv_setpoint',
                'result_json["pv_voltage"] = pv_voltage',
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_count"] = int(np.sum(bus_v > threshold))',
            ]
        )
    else:
        lines.extend(
            [
                "slack_bus = int(ssa.Slack.bus.v[0])",
                "slack_pos = int(np.where(bus_ids == slack_bus)[0][0])",
                "slack_voltage = round(float(bus_v[slack_pos]), 6)",
                'result_json["pv_setpoint"] = current_pv_setpoint',
                'result_json["opened_line_id"] = opened_line_id',
                'result_json["opened_line_bus_pair"] = [opened_bus1, opened_bus2]',
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )

    lines.extend(["", 'print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))'])
    return "\n".join(lines)


def build_structured_n1_screening_script(report_kind: str, state: StructuredAndesState) -> str:
    target_bus = state.target_pq_bus
    scale_factor = state.target_pq_scale_factor
    candidate_pairs = state.n1_candidate_lines
    if target_bus is None or scale_factor is None or not candidate_pairs:
        raise ValueError("N-1 structured codegen requires a targeted PQ edit and candidate lines.")

    required_dependencies = ["andes", "numpy"]
    imports = ["import json", "import andes", "import numpy as np"]
    if state.case_source == "uploaded":
        imports.append("import os")

    lines: List[str] = [
        f"# required_dependencies: {','.join(required_dependencies)}",
        *imports,
        "",
    ]

    if state.case_source == "uploaded":
        lines.extend(
            [
                "script_dir = os.getcwd()",
                f'case = os.path.join(script_dir, "{state.case_reference}")',
                "",
                "def _load_case():",
                '    return andes.load(case, setup=False, no_output=True, log=False)',
            ]
        )
    else:
        lines.extend(
            [
                "def _load_case():",
                "    return andes.load(",
                f'        andes.get_case("{state.case_reference}"),',
                "        setup=False,",
                "        no_output=True,",
                "        log=False,",
                "    )",
            ]
        )

    lines.extend(
        [
            "",
            "def _resolve_line(ssa, bus1, bus2):",
            "    line_bus1 = np.asarray(ssa.Line.bus1.v, dtype=int)",
            "    line_bus2 = np.asarray(ssa.Line.bus2.v, dtype=int)",
            "    matches = np.where(((line_bus1 == bus1) & (line_bus2 == bus2)) | ((line_bus1 == bus2) & (line_bus2 == bus1)))[0]",
            "    if matches.size == 0:",
            '        raise ValueError(f\"No line found for bus pair {bus1}-{bus2}\")',
            "    line_pos = int(matches[0])",
            "    return line_pos, ssa.Line.idx.v[line_pos]",
            "",
            "def _contingency_record(ssa, converged, bus1, bus2, line_idx_value):",
            "    island_sets_raw = ssa.Bus.island_sets if hasattr(ssa.Bus, 'island_sets') else []",
            "    island_sets = list(island_sets_raw or [])",
            "    no_slack_raw = ssa.Bus.nosw_island if hasattr(ssa.Bus, 'nosw_island') else []",
            "    no_slack_islands = int(len(no_slack_raw or []))",
            "    multi_slack_islands = int(len(getattr(ssa.Bus, 'msw_island', []) or []))",
            "    islanded_bus_count = int(ssa.Bus.n_islanded_buses if hasattr(ssa.Bus, 'n_islanded_buses') else 0)",
            "    if no_slack_islands > 0:",
            "        outage_status = 'no_slack_island'",
            "    elif multi_slack_islands > 0:",
            "        outage_status = 'multi_slack_island'",
            "    elif not converged:",
            "        outage_status = 'not_converged'",
            "    elif len(island_sets) > 1 or islanded_bus_count > 0:",
            "        outage_status = 'converged_with_islanding'",
            "    else:",
            "        outage_status = 'converged'",
            "    bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)",
            "    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
            "    min_pos = int(np.argmin(bus_v)) if bus_v.size else 0",
            "    last_mismatch = None",
            "    mismatch_series = getattr(ssa.PFlow, 'mis', None)",
            "    if mismatch_series is not None and len(mismatch_series):",
            "        last_mismatch = round(float(mismatch_series[-1]), 6)",
            "    return {",
            "        'line_id': str(line_idx_value),",
            "        'bus_pair': [int(bus1), int(bus2)],",
            "        'outage_status': outage_status,",
            "        'exit_code': int(getattr(ssa, 'exit_code', 1)),",
            "        'island_count': int(len(island_sets)),",
            "        'no_slack_islands': no_slack_islands,",
            "        'multi_slack_islands': multi_slack_islands,",
            "        'islanded_bus_count': islanded_bus_count,",
            "        'min_bus': int(bus_ids[min_pos]) if bus_v.size else None,",
            "        'min_voltage': round(float(bus_v[min_pos]), 6) if bus_v.size else None,",
            "        'last_mismatch': last_mismatch,",
            "    }",
            "",
            "def _priority_tuple(record):",
            "    severity = {",
            "        'no_slack_island': 4,",
            "        'multi_slack_island': 3,",
            "        'not_converged': 2,",
            "        'converged_with_islanding': 1,",
            "        'converged': 0,",
            "    }.get(record['outage_status'], 0)",
            "    min_voltage = float(record['min_voltage']) if record['min_voltage'] is not None else float('inf')",
            "    last_mismatch = float(record['last_mismatch']) if record['last_mismatch'] is not None else -1.0",
            "    return (",
            "        severity,",
            "        int(record['no_slack_islands']),",
            "        int(record['multi_slack_islands']),",
            "        int(record['island_count']),",
            "        int(record['islanded_bus_count']),",
            "        -min_voltage,",
            "        last_mismatch,",
            "    )",
            "",
            f"target_pq_bus = {target_bus}",
            f"scale_factor = {scale_factor}",
            f"candidate_pairs = {[(bus1, bus2) for bus1, bus2 in candidate_pairs]}",
            "candidate_line_ids = []",
            "worst_record = None",
            "",
            "for bus1, bus2 in candidate_pairs:",
            "    ssa = _load_case()",
            "    ssa.setup()",
            "    pq_bus_array = np.asarray(ssa.PQ.bus.v, dtype=int)",
            "    pq_positions = np.where(pq_bus_array == target_pq_bus)[0]",
            "    if pq_positions.size == 0:",
            '        raise ValueError(f\"No PQ device found at bus {target_pq_bus}\")',
            "    pq_pos = int(pq_positions[0])",
            "    target_pq_idx_value = ssa.PQ.idx.v[pq_pos]",
            "    target_p0_before = float(ssa.PQ.p0.v[pq_pos])",
            "    target_q0_before = float(ssa.PQ.q0.v[pq_pos])",
            '    ssa.PQ.set(src=\"p0\", idx=[target_pq_idx_value], attr=\"v\", value=[target_p0_before * scale_factor])',
            '    ssa.PQ.set(src=\"q0\", idx=[target_pq_idx_value], attr=\"v\", value=[target_q0_before * scale_factor])',
            "    ssa.PFlow.run()",
            "    _line_pos, line_idx_value = _resolve_line(ssa, bus1, bus2)",
            "    candidate_line_ids.append(str(line_idx_value))",
            '    ssa.Line.set(src="u", idx=line_idx_value, attr="v", value=0)',
            "    run_result = ssa.PFlow.run()",
            "    try:",
            "        converged = bool(ssa.PFlow.converged)",
            "    except Exception:",
            "        converged = bool(run_result)",
            "    current_record = _contingency_record(ssa, converged, bus1, bus2, line_idx_value)",
            "    if worst_record is None:",
            "        worst_record = current_record",
            "    elif "
            + ("_priority_tuple(current_record) > _priority_tuple(worst_record)" if report_kind == "n1_failure_aware_screening_report" else "current_record['outage_status'] == 'converged' and (worst_record['outage_status'] != 'converged' or float(current_record['min_voltage']) < float(worst_record['min_voltage']))")
            + ":",
            "        worst_record = current_record",
            "",
            "if worst_record is None:",
            "    raise RuntimeError('No contingency results were produced.')",
            "if "
            + ("False" if report_kind == "n1_failure_aware_screening_report" else "worst_record['outage_status'] != 'converged'")
            + ":",
            "    raise RuntimeError('No converged single-line outage produced finite bus voltages.')",
            "",
            "result_json = {}",
            'result_json["scale_factor"] = round(float(scale_factor), 6)',
            'result_json["candidate_line_ids"] = candidate_line_ids',
            'result_json["worst_line_id"] = worst_record["line_id"]',
            'result_json["worst_line_bus_pair"] = worst_record["bus_pair"]',
            'result_json["worst_min_bus"] = worst_record["min_bus"]',
            'result_json["worst_min_voltage"] = worst_record["min_voltage"]',
            "",
            'print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))',
        ]
    )
    if report_kind == "n1_failure_aware_screening_report":
        lines[-2:-2] = [
            'result_json["worst_outage_status"] = worst_record["outage_status"]',
            'result_json["worst_exit_code"] = int(worst_record["exit_code"])',
            'result_json["worst_island_count"] = int(worst_record["island_count"])',
            'result_json["worst_no_slack_islands"] = int(worst_record["no_slack_islands"])',
            'result_json["worst_islanded_bus_count"] = int(worst_record["islanded_bus_count"])',
        ]
    return "\n".join(lines)


def build_structured_andes_script(
    report_kind: str,
    user_context: str,
    state: StructuredAndesState,
) -> str:
    if not state.case_reference:
        raise ValueError("Structured code generation requires a case reference.")

    if report_kind in {"pq_bus_inspection_report", "pq_bus_scale_report", "pq_bus_scale_threshold_report", "pq_line_outage_threshold_report"}:
        return build_structured_targeted_pq_script(report_kind, user_context, state)
    if report_kind in {"pv_bus_inspection_report", "pv_bus_adjust_threshold_report", "pv_line_outage_report"}:
        return build_structured_targeted_pv_script(report_kind, user_context, state)
    if report_kind in {"n1_screening_report", "n1_failure_aware_screening_report"}:
        return build_structured_n1_screening_script(report_kind, state)

    requested_top_k = extract_top_k_from_prompt(user_context)
    bus_top_k = requested_top_k or state.bus_rank_count or 2
    line_top_k = requested_top_k or state.line_rank_count or 2
    threshold_above = extract_high_voltage_threshold(user_context)
    threshold_below = extract_low_voltage_threshold(user_context)
    angle_threshold = extract_first_float(user_context, r"above ([0-9]*\.?[0-9]+) radians")
    plot_filename = extract_plot_filename(user_context)
    plot_style = "bar" if "bar chart" in (user_context or "").lower() or "bar plot" in (user_context or "").lower() else "line"

    needs_plot = "plot_file" in extract_result_json_keys(user_context)
    needs_line_metrics = "selected_line_ids" in extract_result_json_keys(user_context)
    needs_slack = "slack_bus" in extract_result_json_keys(user_context) or "slack_voltage" in extract_result_json_keys(user_context)
    needs_pv = "pv_bus" in extract_result_json_keys(user_context) or state.pv_setpoint is not None
    needs_bus_arrays = True

    required_dependencies = ["andes", "numpy"]
    imports = ["import json", "import andes", "import numpy as np"]
    if state.case_source == "uploaded" or needs_plot:
        imports.append("import os")
    if needs_plot:
        required_dependencies.append("matplotlib")
        imports.append("import matplotlib.pyplot as plt")

    lines: List[str] = [
        f"# required_dependencies: {','.join(required_dependencies)}",
        *imports,
        "",
    ]

    lines.extend(_build_structured_case_load_lines(state))

    lines.append("")
    for add_op in state.add_ops:
        lines.extend(
            [
                "ssa.add(",
                '    "PQ",',
                "    param_dict={",
                f'        "bus": {add_op["bus"]},',
                f'        "idx": "{add_op["idx"]}",',
                f'        "p0": {add_op["p0"]},',
                f'        "q0": {add_op["q0"]},',
                "    },",
                ")",
                "",
            ]
        )

    lines.extend(["ssa.setup()", ""])

    if state.scale_factor is not None:
        lines.extend(
            [
                f"scale_factor = {state.scale_factor}",
                'ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)',
                'ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)',
                "",
            ]
        )

    if state.slack_setpoint is not None:
        lines.extend(
            [
                f"slack_setpoint = {state.slack_setpoint}",
                'ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])',
                "",
            ]
        )

    if state.pv_setpoint is not None:
        lines.extend(
            [
                f"pv_setpoint = {state.pv_setpoint}",
                'ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])',
                "",
            ]
        )

    lines.extend(
        [
            "ssa.PFlow.run()",
            "",
        ]
    )

    if needs_bus_arrays:
        lines.extend(
            [
                "bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)",
                "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
            ]
        )
    if needs_slack or state.slack_setpoint is not None:
        lines.extend(
            [
                "slack_bus = int(ssa.Slack.bus.v[0])",
                "slack_pos = int(np.where(bus_ids == slack_bus)[0][0])",
                "slack_voltage = round(float(bus_v[slack_pos]), 6)",
            ]
        )
    if needs_pv:
        lines.extend(
            [
                "pv_bus = int(ssa.PV.bus.v[0])",
                "pv_pos = int(np.where(bus_ids == pv_bus)[0][0])",
                "pv_voltage = round(float(bus_v[pv_pos]), 6)",
            ]
        )
    if needs_line_metrics:
        lines.extend(
            [
                "line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]",
                "line_metric = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
            ]
        )

    if needs_plot and plot_filename:
        lines.extend(
            [
                "",
                f'plot_file = "{plot_filename}"',
                "plt.figure(figsize=(10, 4))",
            ]
        )
        if plot_style == "bar":
            lines.append('plt.bar(bus_ids, bus_v, width=0.8)')
        else:
            lines.append('plt.plot(bus_ids, bus_v, marker="o")')
        lines.extend(
            [
                'plt.xlabel("Bus ID")',
                'plt.ylabel("Voltage Magnitude (p.u.)")',
                'plt.title("Bus Voltage Profile")',
                'plt.grid(True, alpha=0.3)',
                "plt.tight_layout()",
                "plt.savefig(plot_file, bbox_inches='tight')",
                "plt.close()",
            ]
        )

    lines.extend(["", "result_json = {}"])

    if report_kind == "baseline_high_rank_report":
        lines.extend(
            [
                f"top_k = {bus_top_k}",
                "selected_idx = np.argsort(bus_v)[-top_k:][::-1]",
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]',
                'result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]',
            ]
        )
    elif report_kind == "add_load_threshold_report":
        threshold = threshold_below if threshold_below is not None else 0.95
        lines.extend(
            [
                f"threshold = {threshold}",
                "below_mask = bus_v < threshold",
                'result_json["added_load_idx"] = "' + state.add_ops[-1]["idx"] + '"',
                f'result_json["added_load_bus"] = {state.add_ops[-1]["bus"]}',
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]',
                'result_json["selected_count"] = int(np.sum(below_mask))',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )
    elif report_kind == "scaled_plot_report" or report_kind == "scaled_bar_plot_report":
        lines.extend(
            [
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])',
                'result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
                'result_json["plot_file"] = plot_file',
            ]
        )
    elif report_kind == "baseline_threshold_low_rank_report":
        threshold = threshold_above if threshold_above is not None else 1.02
        lines.extend(
            [
                f"threshold = {threshold}",
                "above_mask = bus_v > threshold",
                "lowest_idx = np.argsort(bus_v)[:2]",
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_bus_ids"] = [int(item) for item in bus_ids[above_mask]]',
                'result_json["selected_count"] = int(np.sum(above_mask))',
                'result_json["lowest_bus_ids"] = [int(bus_ids[i]) for i in lowest_idx]',
                'result_json["lowest_voltages"] = [round(float(bus_v[i]), 6) for i in lowest_idx]',
            ]
        )
    elif report_kind == "slack_adjust_report":
        threshold = threshold_below if threshold_below is not None else 1.0
        lines.extend(
            [
                f"threshold = {threshold}",
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_setpoint"] = round(float(slack_setpoint), 6)',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["selected_count"] = int(np.sum(bus_v < threshold))',
            ]
        )
    elif report_kind == "extremes_report_with_total_pq":
        lines.extend(
            [
                'result_json["added_load_idx"] = "' + state.add_ops[-1]["idx"] + '"',
                'result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])',
                'result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
                'result_json["total_pq_count"] = int(len(ssa.PQ.idx.v))',
            ]
        )
    elif report_kind == "pv_adjust_report":
        threshold = threshold_above if threshold_above is not None else 1.02
        lines.extend(
            [
                f"threshold = {threshold}",
                'result_json["pv_bus"] = pv_bus',
                'result_json["pv_setpoint"] = round(float(pv_setpoint), 6)',
                'result_json["pv_voltage"] = pv_voltage',
                'result_json["selected_count"] = int(np.sum(bus_v > threshold))',
            ]
        )
    elif report_kind == "extremes_report":
        lines.extend(
            [
                'result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])',
                'result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )
    elif report_kind == "baseline_low_rank_report":
        lines.extend(
            [
                f"top_k = {bus_top_k}",
                "selected_idx = np.argsort(bus_v)[:top_k]",
                'result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]',
                'result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]',
            ]
        )
    elif report_kind == "add_load_slack_threshold_report":
        threshold = threshold_below if threshold_below is not None else 0.95
        lines.extend(
            [
                f"threshold = {threshold}",
                "below_mask = bus_v < threshold",
                'result_json["added_load_idx"] = "' + state.add_ops[-1]["idx"] + '"',
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["threshold"] = round(float(threshold), 6)',
                'result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]',
                'result_json["selected_count"] = int(np.sum(below_mask))',
            ]
        )
    elif report_kind == "slack_plot_low_rank_report":
        lines.extend(
            [
                f"top_k = {bus_top_k}",
                "selected_idx = np.argsort(bus_v)[:top_k]",
                'result_json["slack_setpoint"] = round(float(slack_setpoint), 6)',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]',
                'result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]',
                'result_json["plot_file"] = plot_file',
            ]
        )
    elif report_kind == "line_topk_report":
        lines.extend(
            [
                f"top_k = {line_top_k}",
                "selected_idx = np.argsort(line_metric)[-top_k:][::-1]",
                'result_json["selected_line_ids"] = [str(line_ids[i]) for i in selected_idx]',
                'result_json["selected_line_metrics"] = [round(float(line_metric[i]), 6) for i in selected_idx]',
            ]
        )
    elif report_kind == "scaled_line_threshold_report":
        threshold = angle_threshold if angle_threshold is not None else 0.1
        lines.extend(
            [
                f"angle_threshold = {threshold}",
                "selected_mask = line_metric > angle_threshold",
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["angle_threshold"] = round(float(angle_threshold), 6)',
                'result_json["selected_line_ids"] = [str(item) for item, keep in zip(line_ids, selected_mask) if keep]',
                'result_json["selected_count"] = int(np.sum(selected_mask))',
            ]
        )
    elif report_kind == "add_load_voltage_plot_report":
        lines.extend(
            [
                'result_json["added_load_idx"] = "' + state.add_ops[-1]["idx"] + '"',
                'result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])',
                'result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
                'result_json["plot_file"] = plot_file',
            ]
        )
    elif report_kind == "baseline_slack_extremes_report":
        lines.extend(
            [
                'result_json["slack_bus"] = slack_bus',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])',
                'result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)',
                'result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])',
                'result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)',
            ]
        )
    elif report_kind == "slack_line_topk_report":
        lines.extend(
            [
                f"top_k = {line_top_k}",
                "selected_idx = np.argsort(line_metric)[-top_k:][::-1]",
                'result_json["slack_setpoint"] = round(float(slack_setpoint), 6)',
                'result_json["slack_voltage"] = slack_voltage',
                'result_json["selected_line_ids"] = [str(line_ids[i]) for i in selected_idx]',
                'result_json["selected_line_metrics"] = [round(float(line_metric[i]), 6) for i in selected_idx]',
            ]
        )
    elif report_kind == "slack_scaled_line_threshold_report":
        threshold = angle_threshold if angle_threshold is not None else 0.1
        lines.extend(
            [
                f"angle_threshold = {threshold}",
                "selected_mask = line_metric > angle_threshold",
                'result_json["slack_setpoint"] = round(float(slack_setpoint), 6)',
                'result_json["scale_factor"] = round(float(scale_factor), 6)',
                'result_json["angle_threshold"] = round(float(angle_threshold), 6)',
                'result_json["selected_line_ids"] = [str(item) for item, keep in zip(line_ids, selected_mask) if keep]',
                'result_json["selected_count"] = int(np.sum(selected_mask))',
            ]
        )
    else:
        raise ValueError(f"Unsupported structured report kind: {report_kind}")

    lines.extend(
        [
            "",
            'print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))',
        ]
    )
    return "\n".join(lines)
