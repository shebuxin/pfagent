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
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
line_ids = ssa.Line.idx.v
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(bus_ids.tolist().index(slack_bus))
slack_voltage = _round_float(bus_v[slack_index])
top_k = 4
rank_indices = np.argsort(abs_a1)[-top_k:][::-1]
result = {
    "slack_setpoint": _round_float(1.02),
    "slack_voltage": slack_voltage,
    "selected_line_ids": [str(line_ids[i]) for i in rank_indices],
    "selected_line_metrics": [round(float(abs_a1[i]), 6) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))