# required_dependencies: numpy,andes,json
import numpy as np
import andes
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
slack_setpoint = 1.035
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
threshold = 1.01
result = {
    "slack_bus": slack_bus,
    "slack_setpoint": _round_float(1.035),
    "slack_voltage": slack_voltage,
    "selected_count": int(np.sum(bus_v < threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))