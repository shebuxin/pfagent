import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    "verify_ieee14_021.xlsx",
    setup=True,
    no_output=True,
)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.025
mask = bus_v > threshold
low_rank = np.argsort(bus_v)[:2]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
    "lowest_bus_ids": [int(bus_ids[i]) for i in low_rank],
    "lowest_voltages": [_round_float(bus_v[i]) for i in low_rank],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))