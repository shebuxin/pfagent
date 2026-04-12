# required_dependencies: andes,json
import andes
import json
import numpy as np

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    "verify_ieee14_015.xlsx",
    setup=False,
    no_output=True,
)

ssa.PQ.add(
    param_dict={
        "bus": 4,
        "idx": "PQ_VERIFY_015_A",
        "p0": 0.01,
        "q0": 0.006,
    }
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.0
mask = bus_v < threshold
min_index = int(np.argmin(bus_v))
result = {
    "added_load_idx": "PQ_VERIFY_015_A",
    "added_load_bus": 4,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in np.sort(bus_ids[mask])],
    "selected_count": int(np.sum(mask)),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))