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

ssa.add(
    "PQ",
    param_dict={
        "bus": 10,
        "idx": "PQ_VERIFY_048_B",
        "p0": 0.014,
        "q0": 0.009,
    },
)

ssa.setup()
slack_setpoint = 1.02
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
max_index = int(bus_v.argmax())
min_index = int(bus_v.argmin())
result = {
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
result["added_load_idx"] = "PQ_VERIFY_048_B"
result["total_pq_count"] = int(len(ssa.PQ.idx.v))
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```