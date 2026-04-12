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
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
max_index = int(np.argmax(bus_v))
min_index = int(np.argmin(bus_v))
result = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))