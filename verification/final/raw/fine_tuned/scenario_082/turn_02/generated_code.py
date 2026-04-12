import json
import andes
import numpy as np
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_kundur_082.xlsx")
ssa = andes.load(case_path, setup=False, no_output=True)

ssa.add(
    "PQ",
    param_dict={
        "bus": 4,
        "idx": "PQ_VERIFY_082_D",
        "p0": 0.014,
        "q0": 0.008,
    },
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 0.94
mask = bus_v < threshold
result = {
    "added_load_idx": "PQ_VERIFY_082_D",
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))