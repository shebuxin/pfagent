```python
# required_dependencies: andes,json,numpy
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_pjm5_099.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

slack_setpoint = 1.02
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
scale_factor = 1.05
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
line_ids = ssa.Line.idx.v
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(bus_ids.tolist().index(slack_bus))
slack_voltage = _round_float(ssa.Bus.v.v[slack_index])
angle_threshold = 0.12
mask = abs_a1 > angle_threshold
result = {
    "slack_setpoint": _round_float(1.02),
    "scale_factor": _round_float(1.05),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask].tolist()],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```