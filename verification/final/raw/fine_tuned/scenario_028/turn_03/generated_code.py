# required_dependencies: andes,json,numpy,matplotlib
import andes
import json
import numpy as np
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee14_028.xlsx"
ssa = andes.load(case, setup=False, no_output=True)

ssa.add(
    "PQ",
    param_dict={
        "bus": 9,
        "idx": "PQ_VERIFY_028_D",
        "p0": 0.018,
        "q0": 0.012,
    },
)

ssa.setup()
slack_setpoint = 1.035
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
plot_file = "scenario_028_turn3_line.png"
plt.figure(figsize=(10, 4))
plt.plot(bus_ids, bus_v, marker="o", linewidth=1.5)
plt.xticks(rotation=90)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
top_k = 4
rank_indices = np.argsort(bus_v)[:top_k]
result = {
    "slack_setpoint": _round_float(1.035),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": [int(bus_ids[i]) for i in rank_indices],
    "selected_voltages": [_round_float(bus_v[i]) for i in rank_indices],
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))