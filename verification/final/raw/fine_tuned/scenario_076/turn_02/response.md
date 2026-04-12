```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

case = 'verify_kundur_076.xlsx'
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.Slack.set(src='v0', idx=[ssa.Slack.idx.v[0]], attr='v', value=[1.0])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 0.95
result = {
    'slack_bus': slack_bus,
    'slack_setpoint': 1.0,
    'slack_voltage': slack_voltage,
    'selected_count': int(np.sum(bus_v < threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```