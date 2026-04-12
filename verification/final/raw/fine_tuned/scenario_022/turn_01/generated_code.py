# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee14_022.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.03
mask = bus_v > threshold
top_k = 2
rank_indices = np.argsort(bus_v)[:top_k]
lowest_ranked = [(int(bus_ids[i]), _round_float(bus_v[i])) for i in rank_indices]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
    "lowest_bus_ids": [item[0] for item in lowest_ranked],
    "lowest_voltages": [item[1] for item in lowest_ranked],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))