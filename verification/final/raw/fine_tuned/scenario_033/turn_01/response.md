```python
import andes
import json
import numpy as np

def _round_float(value):
    return round(float(np.asarray(value).item()), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.02
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
```