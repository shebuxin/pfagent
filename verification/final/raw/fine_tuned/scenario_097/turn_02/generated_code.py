# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    "verify_pjm5_097.json",
    setup=True,
    no_output=True,
)

slack_setpoint = 1.0
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
line_bus1 = np.asarray(ssa.Line.bus1.v)
line_bus2 = np.asarray(ssa.Line.bus2.v)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
bus_angle = np.asarray(ssa.Bus.a.v, dtype=float)
top_k = 2
abs_a1 = np.abs(bus_angle[np.searchsorted(bus_ids, line_bus1)])
abs_a2 = np.abs(bus_angle[np.searchsorted(bus_ids, line_bus2)])
abs_diff = np.abs(abs_a1 - abs_a2)
rank_indices = np.argsort(abs_diff)[-top_k:][::-1]
result = {
    "slack_setpoint": _round_float(1.0),
    "slack_voltage": slack_voltage,
    "selected_line_ids": [int(line_ids[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(abs_diff[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))