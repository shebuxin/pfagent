```python
# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_ieee39_044.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = bus_ids.tolist().index(slack_bus)
slack_voltage = _round_float(bus_v[slack_index])
top_k = 3
rank_indices = sorted(range(len(bus_v)), key=lambda i: bus_v[i], reverse=True)[:top_k]
result = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": [int(bus_ids[i]) for i in rank_indices],
    "selected_voltages": [_round_float(bus_v[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```