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
slack_setpoint = 1.02
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(bus_ids.tolist().index(slack_bus))
slack_voltage = _round_float(bus_v[slack_index])
threshold = 0.96
result = {
    "slack_bus": slack_bus,
    "slack_setpoint": _round_float(1.02),
    "slack_voltage": slack_voltage,
    "selected_count": int((bus_v < threshold).sum()),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```