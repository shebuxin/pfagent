# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

case = "verify_kundur_074.xlsx"
ssa = andes.load(case, setup=False, no_output=True)

ssa.add("PQ", param_dict=dict(bus=9, idx="PQ_VERIFY_074_A", p0=0.019, q0=0.012))
ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 0.97
mask = bus_v < threshold
min_index = int(np.argmin(bus_v))
result = {
    "added_load_idx": "PQ_VERIFY_074_A",
    "added_load_bus": 9,
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))