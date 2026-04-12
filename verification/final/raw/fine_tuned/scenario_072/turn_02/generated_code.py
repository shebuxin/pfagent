# required_dependencies: andes,json
import andes
import json
import numpy as np

def _round_float(value):
    return round(float(value), 6)

case = "verify_kundur_072.xlsx"
ssa = andes.load(case, setup=False, no_output=True)

ssa.add(
    "PQ",
    param_dict={
        "bus": 6,
        "idx": "PQ_VERIFY_072_A",
        "p0": 0.013,
        "q0": 0.008,
    },
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 0.95
mask = bus_v < threshold
min_index = int(np.argmin(bus_v))
result = {
    "added_load_idx": "PQ_VERIFY_072_A",
    "added_load_bus": 6,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in np.sort(bus_ids[mask])],
    "selected_count": int(np.sum(mask)),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))