```python
# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_kundur_078.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.set(src="Slack", attr="v0", idx=[ssa.Slack.idx.v[0]], value=[1.02])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_setpoint = _round_float(1.02)
slack_voltage = _round_float(bus_v[slack_index])
threshold = 0.97
result = {
    "slack_bus": slack_bus,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_count": int(np.sum(bus_v < threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```