# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
top_k = 3
rank_indices = np.argsort(bus_v)[-top_k:][::-1]
result = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": [int(bus_ids[i]) for i in rank_indices],
    "selected_voltages": [_round_float(bus_v[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))