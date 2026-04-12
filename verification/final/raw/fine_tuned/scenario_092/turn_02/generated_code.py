# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
)

ssa.setup()
slack_setpoint = 1.03
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
line_bus1 = np.asarray(ssa.Line.bus1.v)
line_bus2 = np.asarray(ssa.Line.bus2.v)
line_a1 = np.asarray(ssa.Line.a1.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
top_k = 5
mask = np.isin(line_bus1, bus_ids)
candidate_ids = line_ids[mask]
candidate_a1 = line_a1[mask]
candidate_bus1 = line_bus1[mask]
candidate_bus2 = line_bus2[mask]
bus_index_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
candidate_a1 = np.asarray([abs(candidate_a1[i] * (bus_v[bus_index_map[candidate_bus1[i]]] - bus_v[bus_index_map[candidate_bus2[i]]])) for i in range(len(candidate_ids))])
rank_indices = np.argsort(candidate_a1)[-top_k:][::-1]
result = {
    "slack_setpoint": _round_float(1.03),
    "slack_voltage": slack_voltage,
    "selected_line_ids": [int(candidate_ids[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(candidate_a1[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))