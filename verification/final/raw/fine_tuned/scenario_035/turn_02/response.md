```python
# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=True,
    no_output=True,
)

ssa.PFlow.set(src="Slack", idx=[ssa.Slack.idx.v[0]], attr="v0", value=[1.03])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_setpoint = _round_float(1.03)
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