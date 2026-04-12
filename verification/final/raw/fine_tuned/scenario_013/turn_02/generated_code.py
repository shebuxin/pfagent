# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 5,
        "idx": "PQ_VERIFY_013_D",
        "p0": 0.016,
        "q0": 0.01,
    },
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 1.005
mask = bus_v < threshold
result = {
    "added_load_idx": "PQ_VERIFY_013_D",
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))