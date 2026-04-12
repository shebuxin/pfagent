```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
)

ssa.setup()
slack_setpoint = 1.0
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
angle_threshold = 0.08
mask = abs_a1 > angle_threshold
result = {
    "slack_setpoint": _round_float(1.0),
    "scale_factor": _round_float(1.03),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```