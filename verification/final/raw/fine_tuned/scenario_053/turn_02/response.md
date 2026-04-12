```python
# required_dependencies: andes,json,numpy
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_ieee39_053.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.015])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
pv_bus = ssa.PV.bus.v[0]
pv_index = int(np.where(bus_ids == pv_bus)[0][0])
threshold = 1.04
result = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": _round_float(1.015),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int(np.sum(bus_v > threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```