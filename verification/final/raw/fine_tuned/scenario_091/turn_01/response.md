```python
# required_dependencies: andes,json
import andes
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(ssa.Bus.idx.v.tolist().index(slack_bus))
slack_voltage = _round_float(ssa.Bus.v.v[slack_index])
max_index = int(bus_v.argmax())
min_index = int(bus_v.argmin())
result = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```