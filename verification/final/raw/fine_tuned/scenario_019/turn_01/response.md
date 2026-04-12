```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee14_019.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.015
mask = bus_v > threshold
top_k = 2
rank_indices = np.argsort(bus_v)[:top_k]
lowest_bus_ids = [int(bus_ids[i]) for i in rank_indices]
lowest_voltages = [_round_float(bus_v[i]) for i in rank_indices]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(np.sum(mask)),
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```