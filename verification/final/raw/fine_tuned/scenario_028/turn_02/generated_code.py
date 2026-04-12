# required_dependencies: andes,json
import andes
import json
import numpy as np

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
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 1.01
mask = bus_v < threshold
result = {
    "added_load_idx": "PQ_VERIFY_028_D",
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))