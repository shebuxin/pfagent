"""Deterministic fallback responses for high-frequency ANDES prompts.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. When the
LLM path fails or is bypassed, these templates emit a manual-aligned
code block (or a short prose explanation) for a small set of recognized
prompt shapes:

- IEEE-14 slack / top-3 bus voltages
- IEEE-39 PQ-load inspection at a requested bus
- Kundur add-one-PQ-load with voltage-bound check
- uploaded-case max/min voltages
- uploaded-case voltage-profile plot
- N-1 single-line outage with voltage-delta plot
- generic branch active-power-flow plot

Each branch returns ``""`` when its keyword pattern is not matched, so
callers can chain ``build_andes_fallback_response(...) or <next>``.

Dependencies (already extracted in earlier Stage 1 batches):
  - src.andes_code.extractors: extract_effective_user_context,
    extract_uploaded_files_from_context, infer_requested_builtin_case,
    extract_requested_bus_number, _extract_voltage_bounds

Known duplication (left intact for byte-exact parity with the
pre-refactor output): the two N-1 outage template branches
(uploaded-case vs builtin-case) share an identical ``_outage_status``
body inside their f-string templates. Deduplication will need to
rewrite the template structure and belongs to a later cleanup pass.
"""

from __future__ import annotations

import os

from src.andes_code.extractors import (
    _extract_voltage_bounds,
    extract_effective_user_context,
    extract_requested_bus_number,
    extract_uploaded_files_from_context,
    infer_requested_builtin_case,
)


def build_andes_fallback_response(user_context: str) -> str:
    """Return a manual-aligned template for a few high-frequency power-flow tasks."""
    effective_user_context = extract_effective_user_context(user_context)
    normalized_prompt = (effective_user_context or user_context or "").lower()
    uploaded_files = extract_uploaded_files_from_context(user_context)
    uploaded_case = os.path.basename(uploaded_files[0]) if uploaded_files else ""
    builtin_case = infer_requested_builtin_case(effective_user_context or user_context)

    if (
        ("ieee 14" in normalized_prompt or "ieee14" in normalized_prompt)
        and "slack bus voltage" in normalized_prompt
        and ("top-3" in normalized_prompt or "top 3" in normalized_prompt)
    ):
        return """```python
# required_dependencies: andes,numpy
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()

bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
bus_id = np.asarray(ssa.Bus.idx.v, dtype=int)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_id == slack_bus)[0][0])

print(f"Slack Bus {slack_bus}: {bus_v[slack_pos]:.4f} p.u.")
print("Top 3 highest bus voltages:")
for i in np.argsort(bus_v)[-3:][::-1]:
    print(f"Bus {int(bus_id[i])}: {bus_v[i]:.4f} p.u.")
```"""

    target_bus = extract_requested_bus_number(effective_user_context or user_context)
    if (
        target_bus
        and ("ieee 39" in normalized_prompt or "ieee39" in normalized_prompt)
        and "pq load" in normalized_prompt
        and "bus voltage" in normalized_prompt
    ):
        return f"""```python
# required_dependencies: andes,numpy
import andes
import numpy as np

target_bus = {int(target_bus)}

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pq_buses = np.asarray(ssa.PQ.bus.v, dtype=int)

bus_positions = np.where(bus_ids == target_bus)[0]
if len(bus_positions) == 0:
    raise ValueError(f"Bus {{target_bus}} was not found in the case.")

bus_pos = int(bus_positions[0])
has_pq = target_bus in pq_buses

if has_pq:
    print(f"PQ load found at bus {{target_bus}}. Bus {{target_bus}} voltage: {{bus_v[bus_pos]:.4f}} p.u.")
else:
    print(f"No PQ load found at bus {{target_bus}}. Bus {{target_bus}} voltage: {{bus_v[bus_pos]:.4f}} p.u.")
```"""

    if "kundur_full" in normalized_prompt and "add one new pq load" in normalized_prompt:
        v_min, v_max = _extract_voltage_bounds(user_context)
        return f"""```python
# required_dependencies: andes
import andes

V_MIN = {v_min}
V_MAX = {v_max}

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.add(
    "PQ",
    param_dict={{
        "bus": 8,
        "idx": "PQ_NEW_1",
        "p0": 0.01,
        "q0": 0.01,
    }},
)

ssa.setup()
ssa.PFlow.run()

violations = [
    (int(bus), float(v))
    for bus, v in zip(ssa.Bus.idx.v, ssa.Bus.v.v)
    if v < V_MIN or v > V_MAX
]

if not violations:
    print("No voltage violations found.")
else:
    print("Voltage violations:")
    for bus, v in violations:
        print(f"Bus {{bus}}: {{v:.4f}} p.u.")
```"""

    if uploaded_case and (
        "maximum and minimum voltages" in normalized_prompt
        or ("maximum voltage" in normalized_prompt and "minimum voltage" in normalized_prompt)
    ):
        return f"""```python
# required_dependencies: andes,numpy
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "{uploaded_case}")
if not os.path.exists(case):
    raise FileNotFoundError(f"Missing uploaded case file: {{case}}")

ssa = andes.load(case, setup=True, no_output=True, log=False)
ssa.PFlow.run()

bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
bus_id = np.asarray(ssa.Bus.idx.v, dtype=int)
max_i = int(np.argmax(bus_v))
min_i = int(np.argmin(bus_v))

print(f"Highest voltage bus: {{int(bus_id[max_i])}}, V={{bus_v[max_i]:.4f}} p.u.")
print(f"Lowest voltage bus: {{int(bus_id[min_i])}}, V={{bus_v[min_i]:.4f}} p.u.")
```"""

    if uploaded_case and "plot" in normalized_prompt and "voltage profile" in normalized_prompt:
        return f"""```python
# required_dependencies: andes,numpy,matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "{uploaded_case}")
if not os.path.exists(case):
    raise FileNotFoundError(f"Missing uploaded case file: {{case}}")

ssa = andes.load(case, setup=True, no_output=True, log=False)
ssa.PFlow.run()

bus_id = np.asarray(ssa.Bus.idx.v, dtype=float)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

plt.figure(figsize=(10, 4))
plt.plot(bus_id, bus_v, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```"""

    if (
        any(
            token in normalized_prompt
            for token in (
                "trip one line",
                "trip a line",
                "trip the line",
                "open one line",
                "open a line",
                "open the line",
                "line outage",
            )
        )
        and "bus voltage change" in normalized_prompt
    ):
        if uploaded_case:
            return f"""```python
# required_dependencies: andes,numpy,matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "{uploaded_case}")
if not os.path.exists(case):
    raise FileNotFoundError(f"Missing uploaded case file: {{case}}")

base_ssa = andes.load(case, setup=True, no_output=True, log=False)
base_ssa.PFlow.run()

bus_ids = np.asarray(base_ssa.Bus.idx.v, dtype=int)
bus_v_before = np.asarray(base_ssa.Bus.v.v, dtype=float)
line_ids = [str(item) for item in np.asarray(base_ssa.Line.idx.v)]

def _outage_status(ssa, converged):
    island_sets = list(getattr(ssa.Bus, "island_sets", []) or [])
    no_slack_islands = int(len(getattr(ssa.Bus, "nosw_island", []) or []))
    islanded_bus_count = int(getattr(ssa.Bus, "n_islanded_buses", 0) or 0)
    if no_slack_islands > 0:
        status = "no_slack_island"
    elif not converged:
        status = "not_converged"
    elif len(island_sets) > 1 or islanded_bus_count > 0:
        status = "converged_with_islanding"
    else:
        status = "converged"
    return status, len(island_sets), no_slack_islands, islanded_bus_count

best_line = None
best_delta = None
best_max_delta = None
screened_lines = 0
rejected_lines = []

for candidate_line in line_ids:
    contingency_ssa = andes.load(case, setup=True, no_output=True, log=False)
    contingency_ssa.PFlow.run()
    contingency_ssa.Line.set(src="u", idx=candidate_line, attr="v", value=0)
    converged = bool(contingency_ssa.PFlow.run())
    contingency_bus_v = np.asarray(contingency_ssa.Bus.v.v, dtype=float)
    status, island_count, no_slack_islands, islanded_bus_count = _outage_status(contingency_ssa, converged)
    screened_lines += 1
    if status != "converged" or not np.all(np.isfinite(contingency_bus_v)):
        rejected_lines.append((candidate_line, status, island_count, no_slack_islands, islanded_bus_count))
        continue
    voltage_delta = np.abs(contingency_bus_v - bus_v_before)
    max_delta = float(np.max(voltage_delta))
    if best_max_delta is None or max_delta > best_max_delta:
        best_line = candidate_line
        best_delta = voltage_delta
        best_max_delta = max_delta

if best_line is None or best_delta is None:
    raise RuntimeError("No converged single-line outage without islanding produced finite bus voltages.")

plt.figure(figsize=(10, 4))
plt.plot(bus_ids, best_delta, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("|ΔV| (p.u.)")
plt.title(f"Bus Voltage Change After Tripping {{best_line}}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Opened line: {{best_line}}")
print(f"Maximum |ΔV| across buses: {{best_max_delta:.6f}} p.u.")
print(f"Screened lines: {{screened_lines}}")
print(f"Rejected contingencies: {{len(rejected_lines)}}")
```"""
        if builtin_case:
            return f"""```python
# required_dependencies: andes,numpy,matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

case = andes.get_case("{builtin_case}")

base_ssa = andes.load(case, setup=True, no_output=True, log=False)
base_ssa.PFlow.run()

bus_ids = np.asarray(base_ssa.Bus.idx.v, dtype=int)
bus_v_before = np.asarray(base_ssa.Bus.v.v, dtype=float)
line_ids = [str(item) for item in np.asarray(base_ssa.Line.idx.v)]

def _outage_status(ssa, converged):
    island_sets = list(getattr(ssa.Bus, "island_sets", []) or [])
    no_slack_islands = int(len(getattr(ssa.Bus, "nosw_island", []) or []))
    islanded_bus_count = int(getattr(ssa.Bus, "n_islanded_buses", 0) or 0)
    if no_slack_islands > 0:
        status = "no_slack_island"
    elif not converged:
        status = "not_converged"
    elif len(island_sets) > 1 or islanded_bus_count > 0:
        status = "converged_with_islanding"
    else:
        status = "converged"
    return status, len(island_sets), no_slack_islands, islanded_bus_count

best_line = None
best_delta = None
best_max_delta = None
screened_lines = 0
rejected_lines = []

for candidate_line in line_ids:
    contingency_ssa = andes.load(case, setup=True, no_output=True, log=False)
    contingency_ssa.PFlow.run()
    contingency_ssa.Line.set(src="u", idx=candidate_line, attr="v", value=0)
    converged = bool(contingency_ssa.PFlow.run())
    contingency_bus_v = np.asarray(contingency_ssa.Bus.v.v, dtype=float)
    status, island_count, no_slack_islands, islanded_bus_count = _outage_status(contingency_ssa, converged)
    screened_lines += 1
    if status != "converged" or not np.all(np.isfinite(contingency_bus_v)):
        rejected_lines.append((candidate_line, status, island_count, no_slack_islands, islanded_bus_count))
        continue
    voltage_delta = np.abs(contingency_bus_v - bus_v_before)
    max_delta = float(np.max(voltage_delta))
    if best_max_delta is None or max_delta > best_max_delta:
        best_line = candidate_line
        best_delta = voltage_delta
        best_max_delta = max_delta

if best_line is None or best_delta is None:
    raise RuntimeError("No converged single-line outage without islanding produced finite bus voltages.")

plt.figure(figsize=(10, 4))
plt.plot(bus_ids, best_delta, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("|ΔV| (p.u.)")
plt.title(f"Bus Voltage Change After Tripping {{best_line}}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Opened line: {{best_line}}")
print(f"Maximum |ΔV| across buses: {{best_max_delta:.6f}} p.u.")
print(f"Screened lines: {{screened_lines}}")
print(f"Rejected contingencies: {{len(rejected_lines)}}")
```"""

    if (
        "plot" in normalized_prompt
        and any(
            token in normalized_prompt
            for token in (
                "branch flow",
                "line flow",
                "branch active power",
                "active power flow of all the branches",
                "active power of all the branches",
            )
        )
    ):
        if uploaded_case:
            return f"""```python
# required_dependencies: andes,numpy,matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "{uploaded_case}")
if not os.path.exists(case):
    raise FileNotFoundError(f"Missing uploaded case file: {{case}}")

ssa = andes.load(case, setup=True, no_output=True, log=False)
ssa.PFlow.run()

line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
p_ij = np.asarray(ssa.Line.a1.e, dtype=float)
p_ji = np.asarray(ssa.Line.a2.e, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(line_ids, p_ij, marker="o", label="Pij")
plt.plot(line_ids, p_ji, marker="x", label="Pji")
plt.xlabel("Line ID")
plt.ylabel("Active Power Flow (p.u.)")
plt.title("Branch Active Power Flow")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```"""
        if builtin_case:
            return f"""```python
# required_dependencies: andes,numpy,matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

ssa = andes.load(
    andes.get_case("{builtin_case}"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()

line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
p_ij = np.asarray(ssa.Line.a1.e, dtype=float)
p_ji = np.asarray(ssa.Line.a2.e, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(line_ids, p_ij, marker="o", label="Pij")
plt.plot(line_ids, p_ji, marker="x", label="Pji")
plt.xlabel("Line ID")
plt.ylabel("Active Power Flow (p.u.)")
plt.title("Branch Active Power Flow")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```"""

    return ""


def build_andes_explanation_fallback_response(user_context: str) -> str:
    """Return a short prose explanation for high-frequency conceptual follow-ups."""
    effective_user_context = extract_effective_user_context(user_context)
    normalized_prompt = (effective_user_context or user_context or "").lower()

    if (
        ("why" in normalized_prompt or "explain" in normalized_prompt)
        and "voltage distribution" in normalized_prompt
        and (
            "tripp" in normalized_prompt
            or "trpping" in normalized_prompt
            or "line trip" in normalized_prompt
            or "line outage" in normalized_prompt
        )
        and "line" in normalized_prompt
    ):
        return (
            "A line trip can leave the bus-voltage profile almost unchanged for a few common reasons. "
            "In a meshed system like IEEE 39-bus, the opened line may not be electrically critical because parallel paths absorb the redistribution. "
            "In that case, the biggest effect often shows up in branch flows or angles rather than in bus-voltage magnitudes. "
            "If the script picked an arbitrary line rather than screening several outages, it may also have selected a line with negligible voltage impact. "
            "A better N-1 check is to screen candidate line outages and compare the minimum bus voltage, the worst-voltage bus, or the maximum |ΔV| across buses."
        )

    return ""
