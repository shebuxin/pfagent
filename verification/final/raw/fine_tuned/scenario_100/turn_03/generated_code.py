# required_dependencies: andes,json,numpy
import andes
import json
import numpy as np
import os

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_pjm5_100.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.set(src="Slack", idx=[ssa.Slack.idx.v[0]], attr="v0", value=[1.03])
ssa.PFlow.set(src="PQ", idx=ssa.PQ.idx.v, attr="p0", value=1.06 * ssa.PQ.p0.v)
ssa.PFlow.set(src="PQ", idx=ssa.PQ.idx.v, attr="q0", value=1.06 * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
line_ids = ssa.Line.idx.v
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
angle_threshold = 0.15
mask = abs_a1 > angle_threshold
result = {
    "slack_setpoint": 1.03,
    "scale_factor": 1.06,
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))