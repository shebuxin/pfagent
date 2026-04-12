```python
# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_ieee14_020.xlsx")
ssa = andes.load(case, setup=False, no_output=True)

ssa.add(
    "PQ",
    param_dict={
        "bus": 5,
        "idx": "PQ_VERIFY_020_B",
        "p0": 0.014,
        "q0": 0.009,
    },
)

ssa.setup()
slack_setpoint = 1.025
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
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
result["added_load_idx"] = "PQ_VERIFY_020_B"
result["total_pq_count"] = int(len(ssa.PQ.idx.v))
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```