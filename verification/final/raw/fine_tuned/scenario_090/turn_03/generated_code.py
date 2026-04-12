# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=True,
    no_output=True,
)

ssa.PFlow.set(src="Slack", idx=[ssa.Slack.idx.v[0]], attr="v0", value=[1.01])
ssa.PFlow.set(src="PQ", idx=ssa.PQ.idx.v, attr="p0", value=1.04 * ssa.PQ.p0.v)
ssa.PFlow.set(src="PQ", idx=ssa.PQ.idx.v, attr="q0", value=1.04 * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
scale_factor = 1.04
angle_threshold = 0.1
mask = abs_a1 > angle_threshold
result = {
    "slack_setpoint": 1.01,
    "scale_factor": scale_factor,
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [int(value) for value in line_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))