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
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(bus_ids.index(slack_bus))
slack_voltage = _round_float(bus_v[slack_index])
line_ids = ssa.Line.idx.v
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
top_k = 5
rank_indices = np.argsort(abs_a1)[-top_k:][::-1]
result = {
    "slack_setpoint": 1.03,
    "slack_voltage": slack_voltage,
    "selected_line_ids": [str(line_ids[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(abs_a1[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))