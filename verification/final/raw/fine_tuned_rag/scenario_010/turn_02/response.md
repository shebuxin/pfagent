```python
# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

pv_setpoint = 1.015
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_pos = int(np.where(bus_ids == pv_bus)[0][0])
pv_voltage = round(float(bus_v[pv_pos]), 6)

result_json = {}
threshold = 1.02
result_json["pv_bus"] = pv_bus
result_json["pv_setpoint"] = round(float(pv_setpoint), 6)
result_json["pv_voltage"] = pv_voltage
result_json["selected_count"] = int(np.sum(bus_v > threshold))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```