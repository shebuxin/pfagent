```python
import json
import andes
import numpy as np
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_kundur_082.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
top_k = 2
rank_indices = np.argsort(bus_v)[:top_k]
result = {
    "selected_bus_ids": [int(bus_ids[i]) for i in rank_indices],
    "selected_voltages": [_round_float(bus_v[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```