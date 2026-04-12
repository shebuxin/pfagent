```python
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
threshold = 1.0
mask = bus_v > threshold
top_k = 2
masked_bus_ids = bus_ids[mask].tolist()
masked_bus_v = bus_v[mask]
rank_indices = np.argsort(masked_bus_v)[:top_k]
lowest_ranked_bus_ids = [masked_bus_ids[i] for i in rank_indices]
lowest_ranked_bus_v = [masked_bus_v[i] for i in rank_indices]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": masked_bus_ids,
    "selected_count": int(np.sum(mask)),
    "lowest_bus_ids": lowest_ranked_bus_ids,
    "lowest_voltages": [_round_float(v) for v in lowest_ranked_bus_v],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```