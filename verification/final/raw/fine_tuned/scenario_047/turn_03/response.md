```python
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee39_047.xlsx"
ssa = andes.load(case, setup=True, no_output=True)

ssa.PQ.add(
    bus=4,
    idx="PQ_VERIFY_047_B",
    p0=0.012,
    q0=0.007,
)

ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[1.015])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
max_index = int(np.argmax(bus_v))
min_index = int(np.argmin(bus_v))
result = {
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
result["added_load_idx"] = "PQ_VERIFY_047_B"
result["total_pq_count"] = int(len(ssa.PQ.idx.v))
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```