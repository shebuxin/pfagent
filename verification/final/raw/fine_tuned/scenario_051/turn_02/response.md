```python
# required_dependencies: andes,numpy,json
import andes
import numpy as np
import json

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee39_051.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.005])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_index = int(np.where(bus_ids == pv_bus)[0][0])
threshold = 1.02
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(1.005),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int(np.sum(bus_v > threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```