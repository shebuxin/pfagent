```python
# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = round(float(bus_v[slack_pos]), 6)

result_json = {}
top_k = 3
selected_idx = np.argsort(bus_v)[-top_k:][::-1]
result_json["slack_bus"] = slack_bus
result_json["slack_voltage"] = slack_voltage
result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]
result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```