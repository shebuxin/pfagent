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
from typing import Dict, Iterable, List, Sequence, Tuple

import andes
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
OUTPUT_JSON = DATA_DIR / "verified_training_examples.json"
REPORT_JSON = DATA_DIR / "verified_training_examples.report.json"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    source_case: str
    prompt_label: str
    uploaded_filename: str | None = None

    @property
    def is_uploaded(self) -> bool:
        return self.uploaded_filename is not None

    @property
    def runtime_name(self) -> str:
        return self.uploaded_filename or self.source_case.replace("/", "_")


@dataclass
class Scenario:
    scenario_id: str
    family: str
    case_key: str
    user: str
    assistant: str
    expected_stdout: str
    artifact_files: List[str] = field(default_factory=list)
    uploaded_files: Dict[str, str] = field(default_factory=dict)


PROMPT_PREFIXES = (
    "Write runnable Python code only.",
    "Return only a runnable Python script.",
    "Generate only Python code that I can run as-is.",
    "Provide runnable Python code only.",
)


CASES: Dict[str, CaseSpec] = {
    "builtin_ieee14": CaseSpec(
        key="builtin_ieee14",
        source_case="ieee14/ieee14_full.xlsx",
        prompt_label="the built-in IEEE 14 full case",
    ),
    "builtin_ieee39": CaseSpec(
        key="builtin_ieee39",
        source_case="ieee39/ieee39.xlsx",
        prompt_label="the built-in IEEE 39 case",
    ),
    "builtin_kundur": CaseSpec(
        key="builtin_kundur",
        source_case="kundur/kundur_full.xlsx",
        prompt_label="the built-in kundur_full case",
    ),
    "builtin_pjm5": CaseSpec(
        key="builtin_pjm5",
        source_case="5bus/pjm5bus.json",
        prompt_label="the built-in pjm5bus case",
    ),
    "builtin_gbnetwork": CaseSpec(
        key="builtin_gbnetwork",
        source_case="GBnetwork/GBnetwork.xlsx",
        prompt_label="the built-in GBnetwork case",
    ),
    "builtin_ei33": CaseSpec(
        key="builtin_ei33",
        source_case="ei/EI_33.xlsx",
        prompt_label="the built-in EI_33 case",
    ),
    "uploaded_ieee14": CaseSpec(
        key="uploaded_ieee14",
        source_case="ieee14/ieee14_full.xlsx",
        prompt_label="my uploaded file north_ieee14_case.xlsx in the current working directory",
        uploaded_filename="north_ieee14_case.xlsx",
    ),
    "uploaded_ieee39": CaseSpec(
        key="uploaded_ieee39",
        source_case="ieee39/ieee39.xlsx",
        prompt_label="my uploaded file plant_study_39.xlsx in the current working directory",
        uploaded_filename="plant_study_39.xlsx",
    ),
    "uploaded_kundur": CaseSpec(
        key="uploaded_kundur",
        source_case="kundur/kundur_full.xlsx",
        prompt_label="my uploaded file kundur_ops_case.xlsx in the current working directory",
        uploaded_filename="kundur_ops_case.xlsx",
    ),
    "uploaded_pjm5": CaseSpec(
        key="uploaded_pjm5",
        source_case="5bus/pjm5bus.json",
        prompt_label="my uploaded file pjm5_snapshot.json in the current working directory",
        uploaded_filename="pjm5_snapshot.json",
    ),
    "uploaded_gbnetwork": CaseSpec(
        key="uploaded_gbnetwork",
        source_case="GBnetwork/GBnetwork.xlsx",
        prompt_label="my uploaded file gbnetwork_uploaded.xlsx in the current working directory",
        uploaded_filename="gbnetwork_uploaded.xlsx",
    ),
    "uploaded_ei33": CaseSpec(
        key="uploaded_ei33",
        source_case="ei/EI_33.xlsx",
        prompt_label="my uploaded file ei33_uploaded.xlsx in the current working directory",
        uploaded_filename="ei33_uploaded.xlsx",
    ),
}


def format_number(value: float, min_decimals: int = 0, max_decimals: int = 4) -> str:
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    if "." not in text and min_decimals > 0:
        text += "." + ("0" * min_decimals)
    elif "." in text and min_decimals > 0:
        head, tail = text.split(".", 1)
        text = f"{head}.{tail.ljust(min_decimals, '0')}"
    return text


def prompt_prefix(index: int) -> str:
    return PROMPT_PREFIXES[index % len(PROMPT_PREFIXES)]


def resolve_source_case(case_spec: CaseSpec) -> str:
    return andes.get_case(case_spec.source_case)


def uploaded_file_map(case_spec: CaseSpec) -> Dict[str, str]:
    if not case_spec.is_uploaded:
        return {}
    assert case_spec.uploaded_filename is not None
    return {case_spec.uploaded_filename: case_spec.source_case}


def render_case_loader(case_spec: CaseSpec, setup: bool) -> Tuple[List[str], List[str]]:
    imports = ["import andes"]
    if case_spec.is_uploaded:
        imports.append("import os")
        assert case_spec.uploaded_filename is not None
        loader_lines = [
            f"case = os.path.join(os.getcwd(), {case_spec.uploaded_filename!r})",
            f"ssa = andes.load(case, setup={setup}, no_output=True)",
        ]
    else:
        loader_lines = [
            "ssa = andes.load(",
            f"    andes.get_case({case_spec.source_case!r}),",
            f"    setup={setup},",
            "    no_output=True,",
            ")",
        ]
    return imports, loader_lines


def render_code(case_spec: CaseSpec, setup: bool, extra_imports: Iterable[str], body_lines: Iterable[str]) -> str:
    imports, loader_lines = render_case_loader(case_spec, setup=setup)
    all_imports = []
    seen = set()
    for entry in [*imports, *extra_imports]:
        if entry not in seen:
            seen.add(entry)
            all_imports.append(entry)

    sections = [
        "\n".join(all_imports),
        "\n".join(loader_lines),
        "\n".join(body_lines).rstrip(),
    ]
    return "\n\n".join(section for section in sections if section.strip()).strip()


def normalize_stdout(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def to_bus_ids(values: Sequence[object]) -> List[str]:
    return [str(value) for value in values]


def load_and_run(case_path: str) -> object:
    ssa = andes.load(case_path, setup=True, no_output=True)
    ssa.PFlow.run()
    return ssa


def load_modify_and_run(case_path: str, modifier) -> object:
    ssa = andes.load(case_path, setup=False, no_output=True)
    modifier(ssa)
    ssa.setup()
    ssa.PFlow.run()
    return ssa


def build_voltage_topk(case_spec: CaseSpec, index: int, k: int, include_slack: bool) -> Scenario:
    ssa = load_and_run(resolve_source_case(case_spec))
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    top_indices = np.argsort(bus_v)[-k:][::-1]

    lines: List[str] = []
    if include_slack:
        lines.append(f"Slack bus voltage: {float(ssa.Slack.v.v[0]):.4f} p.u.")
    for rank, idx in enumerate(top_indices, start=1):
        lines.append(f"Top {rank}: Bus {bus_ids[idx]} voltage = {bus_v[idx]:.4f} p.u.")

    body = [
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        f"top_indices = np.argsort(bus_v)[-{k}:][::-1]",
        "",
    ]
    if include_slack:
        body.append('print(f"Slack bus voltage: {float(ssa.Slack.v.v[0]):.4f} p.u.")')
    body.extend(
        [
            "for rank, idx in enumerate(top_indices, start=1):",
            '    print(f"Top {rank}: Bus {bus_ids[idx]} voltage = {bus_v[idx]:.4f} p.u.")',
        ]
    )

    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, "
        f"print the slack bus voltage, and list the top-{k} highest-voltage buses."
        if include_slack
        else f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and list the top-{k} highest-voltage buses."
    )

    return Scenario(
        scenario_id=f"{case_spec.key}_top{k}_{'with_slack' if include_slack else 'plain'}",
        family="voltage_topk",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout="\n".join(lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_voltage_bottomk(case_spec: CaseSpec, index: int, k: int) -> Scenario:
    ssa = load_and_run(resolve_source_case(case_spec))
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    bottom_indices = np.argsort(bus_v)[:k]
    lines = [
        f"Lowest {rank}: Bus {bus_ids[idx]} voltage = {bus_v[idx]:.4f} p.u."
        for rank, idx in enumerate(bottom_indices, start=1)
    ]

    body = [
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        f"bottom_indices = np.argsort(bus_v)[:{k}]",
        "",
        "for rank, idx in enumerate(bottom_indices, start=1):",
        '    print(f"Lowest {rank}: Bus {bus_ids[idx]} voltage = {bus_v[idx]:.4f} p.u.")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and print the {k} lowest-voltage buses."
    return Scenario(
        scenario_id=f"{case_spec.key}_bottom{k}",
        family="voltage_bottomk",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout="\n".join(lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_count_above(case_spec: CaseSpec, index: int, threshold: float) -> Scenario:
    threshold_label = format_number(threshold, min_decimals=1)
    ssa = load_and_run(resolve_source_case(case_spec))
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    count = int(np.sum(bus_v > threshold))
    expected = f"Buses above {threshold_label} p.u.: {count}"

    body = [
        f"threshold = {threshold_label}",
        "",
        "ssa.PFlow.run()",
        "",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        "count = int(np.sum(bus_v > threshold))",
        'print(f"Buses above {threshold} p.u.: {count}")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and count how many buses are above {threshold_label} p.u."
    return Scenario(
        scenario_id=f"{case_spec.key}_count_above_{threshold_label.replace('.', '_')}",
        family="count_above_threshold",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_list_below(case_spec: CaseSpec, index: int, threshold: float) -> Scenario:
    threshold_label = format_number(threshold, min_decimals=1)
    ssa = load_and_run(resolve_source_case(case_spec))
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    matches = [(bus_id, voltage) for bus_id, voltage in zip(bus_ids, bus_v) if voltage < threshold]

    if matches:
        lines = [f"Buses below {threshold_label} p.u.:"] + [
            f"Bus {bus_id}: {voltage:.4f} p.u." for bus_id, voltage in matches
        ]
    else:
        lines = [f"No buses below {threshold_label} p.u."]

    body = [
        f"threshold = {threshold_label}",
        "",
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        "matches = [(bus_id, voltage) for bus_id, voltage in zip(bus_ids, bus_v) if voltage < threshold]",
        "",
        "if matches:",
        '    print(f"Buses below {threshold} p.u.:")',
        "    for bus_id, voltage in matches:",
        '        print(f"Bus {bus_id}: {voltage:.4f} p.u.")',
        "else:",
        '    print(f"No buses below {threshold} p.u.")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and print every bus below {threshold_label} p.u."
    return Scenario(
        scenario_id=f"{case_spec.key}_below_{threshold_label.replace('.', '_')}",
        family="list_below_threshold",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout="\n".join(lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_minmax(case_spec: CaseSpec, index: int) -> Scenario:
    ssa = load_and_run(resolve_source_case(case_spec))
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    idx_max = int(np.argmax(bus_v))
    idx_min = int(np.argmin(bus_v))

    expected = "\n".join(
        [
            f"Maximum voltage: Bus {bus_ids[idx_max]} = {bus_v[idx_max]:.4f} p.u.",
            f"Minimum voltage: Bus {bus_ids[idx_min]} = {bus_v[idx_min]:.4f} p.u.",
        ]
    )

    body = [
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        "idx_max = int(np.argmax(bus_v))",
        "idx_min = int(np.argmin(bus_v))",
        'print(f"Maximum voltage: Bus {bus_ids[idx_max]} = {bus_v[idx_max]:.4f} p.u.")',
        'print(f"Minimum voltage: Bus {bus_ids[idx_min]} = {bus_v[idx_min]:.4f} p.u.")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and print the maximum-voltage bus together with the minimum-voltage bus."
    return Scenario(
        scenario_id=f"{case_spec.key}_max_min_voltage",
        family="max_min_voltage",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_plot(case_spec: CaseSpec, index: int, plot_kind: str) -> Scenario:
    plot_suffix = "line" if plot_kind == "line" else "bar"
    plot_file = f"{case_spec.runtime_name}_voltage_profile_{plot_suffix}.png"
    ssa = load_and_run(resolve_source_case(case_spec))
    _ = ssa.Bus.v.v  # ensure the case runs during generation

    if plot_kind == "line":
        plot_line = "plt.plot(bus_ids, bus_v, marker='o')"
        prompt_body = "save a line plot of the bus voltage profile."
    else:
        plot_line = "plt.bar(bus_ids, bus_v)"
        prompt_body = "save a bar plot of the bus voltage profile."

    body = [
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = ssa.Bus.v.v",
        "plt.figure(figsize=(10, 4))",
        plot_line,
        "plt.xlabel('Bus')",
        "plt.ylabel('Voltage (p.u.)')",
        "plt.title('Bus Voltage Profile')",
        "plt.tight_layout()",
        f"plot_file = {plot_file!r}",
        "plt.savefig(plot_file, dpi=150)",
        'print(f"Saved plot to {plot_file}")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and {prompt_body}"
    return Scenario(
        scenario_id=f"{case_spec.key}_plot_{plot_suffix}",
        family=f"plot_{plot_suffix}",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import matplotlib.pyplot as plt"], body_lines=body),
        expected_stdout=f"Saved plot to {plot_file}",
        artifact_files=[plot_file],
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_line_angle_topk(case_spec: CaseSpec, index: int, k: int) -> Scenario:
    ssa = load_and_run(resolve_source_case(case_spec))
    line_ids = to_bus_ids(ssa.Line.idx.v)
    angles = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
    top_indices = np.argsort(angles)[-k:][::-1]
    expected_lines = [
        f"Top {rank}: Line {line_ids[idx]} |a1| = {angles[idx]:.6f} rad"
        for rank, idx in enumerate(top_indices, start=1)
    ]

    body = [
        "ssa.PFlow.run()",
        "",
        "line_ids = [str(line_id) for line_id in ssa.Line.idx.v]",
        "angles = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))",
        f"top_indices = np.argsort(angles)[-{k}:][::-1]",
        "",
        "for rank, idx in enumerate(top_indices, start=1):",
        '    print(f"Top {rank}: Line {line_ids[idx]} |a1| = {angles[idx]:.6f} rad")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and print the top-{k} lines by absolute sending-end phase angle."
    return Scenario(
        scenario_id=f"{case_spec.key}_line_angle_top{k}",
        family="line_angle_topk",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout="\n".join(expected_lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_line_angle_threshold(case_spec: CaseSpec, index: int, threshold: float) -> Scenario:
    threshold_label = format_number(threshold, min_decimals=1)
    ssa = load_and_run(resolve_source_case(case_spec))
    line_ids = to_bus_ids(ssa.Line.idx.v)
    angles = np.asarray(ssa.Line.a1.e, dtype=float)
    matches = [(line_id, abs(angle)) for line_id, angle in zip(line_ids, angles) if abs(angle) > threshold]
    if matches:
        expected_lines = [f"Lines above {threshold_label} rad:"] + [
            f"Line {line_id}: |a1| = {angle:.6f} rad" for line_id, angle in matches
        ]
    else:
        expected_lines = [f"No lines above {threshold_label} rad"]

    body = [
        f"threshold = {threshold_label}",
        "",
        "ssa.PFlow.run()",
        "",
        "matches = []",
        "for line_id, angle in zip([str(value) for value in ssa.Line.idx.v], ssa.Line.a1.e):",
        "    if abs(angle) > threshold:",
        "        matches.append((line_id, abs(float(angle))))",
        "",
        "if matches:",
        '    print(f"Lines above {threshold} rad:")',
        "    for line_id, angle in matches:",
        '        print(f"Line {line_id}: |a1| = {angle:.6f} rad")',
        "else:",
        '    print(f"No lines above {threshold} rad")',
    ]
    prompt = f"{prompt_prefix(index)} Use {case_spec.prompt_label}, run power flow, and print every line whose absolute sending-end phase angle is above {threshold_label} radians."
    return Scenario(
        scenario_id=f"{case_spec.key}_line_angle_above_{threshold_label.replace('.', '_')}",
        family="line_angle_threshold",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=[], body_lines=body),
        expected_stdout="\n".join(expected_lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_add_load_violations(
    case_spec: CaseSpec,
    index: int,
    bus: int,
    p0: float,
    q0: float,
    vmin: float,
    vmax: float,
) -> Scenario:
    p0_label = format_number(p0, max_decimals=4)
    q0_label = format_number(q0, max_decimals=4)
    vmin_label = format_number(vmin, min_decimals=1)
    vmax_label = format_number(vmax, min_decimals=1)
    load_idx = f"PQ_EXTRA_{bus}_{p0_label.replace('.', '_')}"

    def modifier(ssa):
        ssa.add("PQ", param_dict=dict(bus=bus, idx=load_idx, p0=p0, q0=q0))

    ssa = load_modify_and_run(resolve_source_case(case_spec), modifier)
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    violations = [(bus_id, voltage) for bus_id, voltage in zip(bus_ids, bus_v) if voltage < vmin or voltage > vmax]
    if violations:
        expected_lines = [f"Voltage violations outside [{vmin_label}, {vmax_label}] p.u.:"] + [
            f"Bus {bus_id}: {voltage:.4f} p.u." for bus_id, voltage in violations
        ]
    else:
        expected_lines = [f"No bus voltage violations outside [{vmin_label}, {vmax_label}] p.u."]

    body = [
        f"ssa.add('PQ', param_dict=dict(bus={bus}, idx={load_idx!r}, p0={p0_label}, q0={q0_label}))",
        "ssa.setup()",
        "ssa.PFlow.run()",
        "",
        f"vmin = {vmin_label}",
        f"vmax = {vmax_label}",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        "violations = [(bus_id, voltage) for bus_id, voltage in zip(bus_ids, bus_v) if voltage < vmin or voltage > vmax]",
        "",
        "if violations:",
        '    print(f"Voltage violations outside [{vmin}, {vmax}] p.u.:")',
        "    for bus_id, voltage in violations:",
        '        print(f"Bus {bus_id}: {voltage:.4f} p.u.")',
        "else:",
        '    print(f"No bus voltage violations outside [{vmin}, {vmax}] p.u.")',
    ]
    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, add one new PQ load at bus {bus} before setup "
        f"with p0={p0_label} and q0={q0_label}, run power flow, and report any buses outside [{vmin_label}, {vmax_label}] p.u."
    )
    return Scenario(
        scenario_id=f"{case_spec.key}_add_load_{bus}_{p0_label.replace('.', '_')}_{q0_label.replace('.', '_')}_violations",
        family="add_load_violations",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=False, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout="\n".join(expected_lines),
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_add_load_min_bus(case_spec: CaseSpec, index: int, bus: int, p0: float, q0: float) -> Scenario:
    p0_label = format_number(p0, max_decimals=4)
    q0_label = format_number(q0, max_decimals=4)
    load_idx = f"PQ_EXTRA_{bus}_{p0_label.replace('.', '_')}"

    def modifier(ssa):
        ssa.add("PQ", param_dict=dict(bus=bus, idx=load_idx, p0=p0, q0=q0))

    ssa = load_modify_and_run(resolve_source_case(case_spec), modifier)
    bus_ids = to_bus_ids(ssa.Bus.idx.v)
    bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
    idx_min = int(np.argmin(bus_v))
    expected = f"Minimum-voltage bus after adding load: Bus {bus_ids[idx_min]} = {bus_v[idx_min]:.4f} p.u."

    body = [
        f"ssa.add('PQ', param_dict=dict(bus={bus}, idx={load_idx!r}, p0={p0_label}, q0={q0_label}))",
        "ssa.setup()",
        "ssa.PFlow.run()",
        "",
        "bus_ids = [str(bus_id) for bus_id in ssa.Bus.idx.v]",
        "bus_v = np.asarray(ssa.Bus.v.v, dtype=float)",
        "idx_min = int(np.argmin(bus_v))",
        'print(f"Minimum-voltage bus after adding load: Bus {bus_ids[idx_min]} = {bus_v[idx_min]:.4f} p.u.")',
    ]
    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, add one new PQ load at bus {bus} before setup "
        f"with p0={p0_label} and q0={q0_label}, run power flow, and print the minimum-voltage bus."
    )
    return Scenario(
        scenario_id=f"{case_spec.key}_add_load_{bus}_{p0_label.replace('.', '_')}_{q0_label.replace('.', '_')}_min",
        family="add_load_min_bus",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=False, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_scale_pq_totals(case_spec: CaseSpec, index: int, scale: float) -> Scenario:
    scale_label = format_number(scale, min_decimals=1)

    def modifier(ssa):
        ssa.setup()
        ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale * ssa.PQ.p0.v)
        ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale * ssa.PQ.q0.v)

    ssa = andes.load(resolve_source_case(case_spec), setup=False, no_output=True)
    modifier(ssa)
    ssa.PFlow.run()
    total_p = float(np.sum(np.asarray(ssa.PQ.p0.v, dtype=float)))
    total_q = float(np.sum(np.asarray(ssa.PQ.q0.v, dtype=float)))
    expected = f"Scaled total PQ load: P={total_p:.6f} p.u., Q={total_q:.6f} p.u."

    body = [
        f"scale = {scale_label}",
        "ssa.setup()",
        "ssa.PQ.set(src='p0', idx=ssa.PQ.idx.v, attr='v', value=scale * ssa.PQ.p0.v)",
        "ssa.PQ.set(src='q0', idx=ssa.PQ.idx.v, attr='v', value=scale * ssa.PQ.q0.v)",
        "ssa.PFlow.run()",
        "",
        "total_p = float(np.sum(np.asarray(ssa.PQ.p0.v, dtype=float)))",
        "total_q = float(np.sum(np.asarray(ssa.PQ.q0.v, dtype=float)))",
        'print(f"Scaled total PQ load: P={total_p:.6f} p.u., Q={total_q:.6f} p.u.")',
    ]
    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, scale every PQ load by a factor of {scale_label}, "
        "run power flow, and print the new total active and reactive PQ load in per unit."
    )
    return Scenario(
        scenario_id=f"{case_spec.key}_scale_pq_{scale_label.replace('.', '_')}",
        family="scale_pq_totals",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=False, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_pv_setpoints(case_spec: CaseSpec, index: int, p_values: Sequence[float], q_values: Sequence[float]) -> Scenario:
    p_literal = ", ".join(format_number(value, max_decimals=4) for value in p_values)
    q_literal = ", ".join(format_number(value, max_decimals=4) for value in q_values)
    idx_literal = "1, 2"

    ssa = andes.load(resolve_source_case(case_spec), setup=False, no_output=True)
    ssa.setup()
    ssa.PV.set(src="p0", idx=[1, 2], attr="v", value=np.asarray(p_values, dtype=float))
    ssa.PV.set(src="q0", idx=[1, 2], attr="v", value=np.asarray(q_values, dtype=float))
    ssa.PFlow.run()
    updated_p = [round(float(value), 4) for value in np.asarray(ssa.PV.p0.v[:2], dtype=float)]
    updated_q = [round(float(value), 4) for value in np.asarray(ssa.PV.q0.v[:2], dtype=float)]
    expected = "\n".join(
        [
            f"Updated PV p0 setpoints: {updated_p}",
            f"Updated PV q0 setpoints: {updated_q}",
        ]
    )

    body = [
        "ssa.setup()",
        f"ssa.PV.set(src='p0', idx=[{idx_literal}], attr='v', value=np.asarray([{p_literal}], dtype=float))",
        f"ssa.PV.set(src='q0', idx=[{idx_literal}], attr='v', value=np.asarray([{q_literal}], dtype=float))",
        "ssa.PFlow.run()",
        "",
        "updated_p = [round(float(value), 4) for value in np.asarray(ssa.PV.p0.v[:2], dtype=float)]",
        "updated_q = [round(float(value), 4) for value in np.asarray(ssa.PV.q0.v[:2], dtype=float)]",
        'print(f"Updated PV p0 setpoints: {updated_p}")',
        'print(f"Updated PV q0 setpoints: {updated_q}")',
    ]
    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, update PV generators 1 and 2 so that "
        f"p0 becomes [{p_literal}] and q0 becomes [{q_literal}], run power flow, and print the updated PV setpoints."
    )
    return Scenario(
        scenario_id=f"{case_spec.key}_pv_setpoints",
        family="pv_setpoints",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=False, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_slack_setpoints(case_spec: CaseSpec, index: int, v0: float, a0: float) -> Scenario:
    v0_label = format_number(v0, max_decimals=4)
    a0_label = format_number(a0, max_decimals=4)

    ssa = andes.load(resolve_source_case(case_spec), setup=True, no_output=True)
    ssa.Slack.set(src="v0", idx=10, attr="v", value=np.asarray([v0], dtype=float))
    ssa.Slack.set(src="a0", idx=10, attr="v", value=np.asarray([a0], dtype=float))
    ssa.PFlow.run()
    updated_v0 = float(np.asarray(ssa.Slack.v0.v, dtype=float)[0])
    updated_a0 = float(np.asarray(ssa.Slack.a0.v, dtype=float)[0])
    expected = f"Updated slack setpoints: v0={updated_v0:.4f}, a0={updated_a0:.4f}"

    body = [
        f"ssa.Slack.set(src='v0', idx=10, attr='v', value=np.asarray([{v0_label}], dtype=float))",
        f"ssa.Slack.set(src='a0', idx=10, attr='v', value=np.asarray([{a0_label}], dtype=float))",
        "ssa.PFlow.run()",
        "",
        "updated_v0 = float(np.asarray(ssa.Slack.v0.v, dtype=float)[0])",
        "updated_a0 = float(np.asarray(ssa.Slack.a0.v, dtype=float)[0])",
        'print(f"Updated slack setpoints: v0={updated_v0:.4f}, a0={updated_a0:.4f}")',
    ]
    prompt = (
        f"{prompt_prefix(index)} Use {case_spec.prompt_label}, update the slack bus setpoints to "
        f"v0={v0_label} and a0={a0_label}, run power flow, and print the updated slack setpoints."
    )
    return Scenario(
        scenario_id=f"{case_spec.key}_slack_setpoints",
        family="slack_setpoints",
        case_key=case_spec.key,
        user=prompt,
        assistant=render_code(case_spec, setup=True, extra_imports=["import numpy as np"], body_lines=body),
        expected_stdout=expected,
        uploaded_files=uploaded_file_map(case_spec),
    )


def build_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []
    idx = 0

    def add(scenario: Scenario) -> None:
        nonlocal idx
        scenarios.append(scenario)
        idx += 1

    for case_key, k, include_slack in [
        ("builtin_ieee14", 3, True),
        ("builtin_ieee39", 2, True),
        ("builtin_ei33", 3, False),
        ("uploaded_ieee14", 2, True),
        ("uploaded_ieee39", 2, True),
        ("uploaded_ei33", 3, False),
    ]:
        add(build_voltage_topk(CASES[case_key], idx, k, include_slack))

    for case_key, k in [
        ("builtin_ieee14", 2),
        ("builtin_ieee39", 2),
        ("builtin_gbnetwork", 3),
        ("uploaded_ieee14", 2),
        ("uploaded_gbnetwork", 3),
    ]:
        add(build_voltage_bottomk(CASES[case_key], idx, k))

    for case_key, threshold in [
        ("builtin_ieee14", 1.02),
        ("builtin_ieee39", 1.04),
        ("builtin_ei33", 1.01),
        ("builtin_gbnetwork", 1.02),
        ("uploaded_ieee14", 1.01),
        ("uploaded_ieee39", 1.03),
        ("uploaded_ei33", 1.01),
    ]:
        add(build_count_above(CASES[case_key], idx, threshold))

    for case_key, threshold in [
        ("builtin_ieee14", 1.0),
        ("builtin_ieee39", 0.95),
        ("builtin_kundur", 0.97),
        ("builtin_gbnetwork", 0.95),
        ("uploaded_ieee14", 1.0),
        ("uploaded_ieee39", 0.96),
        ("uploaded_kundur", 0.97),
    ]:
        add(build_list_below(CASES[case_key], idx, threshold))

    for case_key in [
        "builtin_ieee14",
        "builtin_ieee39",
        "builtin_gbnetwork",
        "uploaded_ieee14",
        "uploaded_ieee39",
        "uploaded_gbnetwork",
    ]:
        add(build_minmax(CASES[case_key], idx))

    for case_key, plot_kind in [
        ("builtin_ieee14", "line"),
        ("builtin_ieee39", "bar"),
        ("builtin_ei33", "line"),
        ("uploaded_ieee14", "bar"),
        ("uploaded_ieee39", "line"),
        ("uploaded_ei33", "bar"),
    ]:
        add(build_plot(CASES[case_key], idx, plot_kind))

    for case_key, k in [
        ("builtin_pjm5", 2),
        ("builtin_pjm5", 3),
        ("uploaded_pjm5", 2),
        ("uploaded_pjm5", 3),
    ]:
        add(build_line_angle_topk(CASES[case_key], idx, k))

    for case_key, threshold in [
        ("builtin_pjm5", 0.08),
        ("builtin_pjm5", 0.1),
        ("uploaded_pjm5", 0.08),
        ("uploaded_pjm5", 0.1),
    ]:
        add(build_line_angle_threshold(CASES[case_key], idx, threshold))

    for case_key, bus, p0, q0, vmin, vmax in [
        ("builtin_kundur", 9, 0.02, 0.015, 0.94, 1.06),
        ("builtin_ieee39", 7, 0.018, 0.009, 0.95, 1.05),
        ("uploaded_kundur", 6, 0.025, 0.02, 0.93, 1.07),
        ("uploaded_ieee39", 7, 0.018, 0.009, 0.95, 1.05),
    ]:
        add(build_add_load_violations(CASES[case_key], idx, bus, p0, q0, vmin, vmax))

    for case_key, bus, p0, q0 in [
        ("builtin_kundur", 4, 0.03, 0.01),
        ("builtin_ieee39", 4, 0.02, 0.01),
        ("uploaded_kundur", 5, 0.02, 0.012),
        ("uploaded_ieee39", 4, 0.02, 0.01),
    ]:
        add(build_add_load_min_bus(CASES[case_key], idx, bus, p0, q0))

    for case_key, scale in [
        ("builtin_ieee39", 1.05),
        ("builtin_ieee14", 1.08),
        ("builtin_gbnetwork", 1.03),
        ("uploaded_ieee39", 1.05),
        ("uploaded_ieee14", 0.95),
        ("uploaded_gbnetwork", 1.03),
    ]:
        add(build_scale_pq_totals(CASES[case_key], idx, scale))

    for case_key, p_values, q_values in [
        ("builtin_ieee39", [5.0, 7.0], [2.0, 2.5]),
        ("uploaded_ieee39", [5.0, 7.0], [2.0, 2.5]),
    ]:
        add(build_pv_setpoints(CASES[case_key], idx, p_values, q_values))

    for case_key, v0, a0 in [
        ("builtin_ieee39", 1.01, 0.02),
        ("uploaded_ieee39", 1.01, 0.02),
    ]:
        add(build_slack_setpoints(CASES[case_key], idx, v0, a0))

    return scenarios


def validate_scenario(scenario: Scenario) -> Dict[str, object]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfagent-verified")

    with tempfile.TemporaryDirectory(prefix=f"pfagent-{scenario.scenario_id}-", dir="/tmp") as tmpdir:
        runtime_dir = Path(tmpdir)

        for filename, source_case in scenario.uploaded_files.items():
            shutil.copyfile(andes.get_case(source_case), runtime_dir / filename)

        script_path = runtime_dir / "scenario.py"
        script_path.write_text(scenario.assistant, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script_path.name)],
            cwd=runtime_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        actual_stdout = normalize_stdout(result.stdout)
        actual_stderr = normalize_stdout(result.stderr)
        stdout_match = actual_stdout == normalize_stdout(scenario.expected_stdout)
        artifacts_present = all((runtime_dir / artifact).exists() for artifact in scenario.artifact_files)
        passed = result.returncode == 0 and stdout_match and artifacts_present

        return {
            "scenario_id": scenario.scenario_id,
            "family": scenario.family,
            "case_key": scenario.case_key,
            "passed": passed,
            "returncode": result.returncode,
            "stdout_match": stdout_match,
            "artifacts_present": artifacts_present,
            "expected_stdout": scenario.expected_stdout,
            "actual_stdout": actual_stdout,
            "stderr": actual_stderr,
            "artifact_files": scenario.artifact_files,
        }


def write_examples(examples: Sequence[Scenario], output_path: Path) -> None:
    payload = {
        "examples": [
            {
                "id": scenario.scenario_id,
                "family": scenario.family,
                "case_key": scenario.case_key,
                "user": scenario.user,
                "assistant": scenario.assistant,
                "validation": {
                    "expected_stdout": scenario.expected_stdout,
                    "artifact_files": scenario.artifact_files,
                    "uploaded_files": scenario.uploaded_files,
                },
            }
            for scenario in examples
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_report(results: Sequence[Dict[str, object]], output_path: Path) -> None:
    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "results": list(results),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strictly validated PFAGENT fine-tuning examples.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_JSON),
        help="Path for the verified examples JSON file.",
    )
    parser.add_argument(
        "--report-output",
        default=str(REPORT_JSON),
        help="Path for the validation report JSON file.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write only passing examples even if some scenarios fail validation.",
    )
    args = parser.parse_args()

    scenarios = build_scenarios()
    results = [validate_scenario(scenario) for scenario in scenarios]
    passed_ids = {item["scenario_id"] for item in results if item["passed"]}
    passed_examples = [scenario for scenario in scenarios if scenario.scenario_id in passed_ids]

    write_examples(passed_examples, Path(args.output))
    write_report(results, Path(args.report_output))

    passed = len(passed_examples)
    total = len(scenarios)
    print(f"Verified examples written to: {args.output}")
    print(f"Validation report written to: {args.report_output}")
    print(f"Validation summary: {passed}/{total} scenarios passed")

    failed = [item for item in results if not item["passed"]]
    for item in failed:
        print(f"- FAIL: {item['scenario_id']}")
        print(f"  returncode={item['returncode']} stdout_match={item['stdout_match']} artifacts_present={item['artifacts_present']}")

    if failed and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
