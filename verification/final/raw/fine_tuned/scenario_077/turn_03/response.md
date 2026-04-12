```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json
import os

def _round_float(value):
    return round(float(value), 6)

wd = os.getcwd()
case = os.path.join(wd, "verify_kundur_077.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 7,
        "idx": "PQ_VERIFY_077_B",
        "p0": 0.016,
        "q0": 0.011,
    },
)

ssa.setup()
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[1.01])
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
result["added_load_idx"] = "PQ_VERIFY_077_B"
result["total_pq_count"] = int(len(ssa.PQ.idx.v))
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```