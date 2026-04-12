```python
# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_ieee39_048.xlsx")
ssa = andes.load(case_path, setup=False, no_output=True)

ssa.setup()
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
threshold = 1.03
mask = bus_v > threshold
low_rank_indices = sorted(range(len(bus_v)), key=lambda i: bus_v[i])[:2]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(mask.sum()),
    "lowest_bus_ids": [int(bus_ids[i]) for i in low_rank_indices],
    "lowest_voltages": [_round_float(bus_v[i]) for i in low_rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```