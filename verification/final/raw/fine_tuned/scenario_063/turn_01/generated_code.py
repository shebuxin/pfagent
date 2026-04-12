import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=True,
    no_output=True,
)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.01
mask = bus_v > threshold
top_k = 2
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
    "lowest_bus_ids": [int(value) for value in bus_ids[np.argsort(bus_v)[:top_k]]],
    "lowest_voltages": [_round_float(value) for value in np.sort(bus_v)[:top_k]],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))