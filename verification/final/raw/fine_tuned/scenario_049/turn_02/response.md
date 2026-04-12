```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_ieee39_049.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

slack_setpoint = 1.03
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 0.97
result = {
    "slack_bus": slack_bus,
    "slack_setpoint": _round_float(1.03),
    "slack_voltage": slack_voltage,
    "selected_count": int(np.sum(bus_v < threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```