```python
# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.01])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
pv_bus = ssa.PV.bus.v[0]
pv_index = int(bus_ids.index(pv_bus))
threshold = 1.01
count = sum(1 for voltage in bus_v if voltage > threshold)
result = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": _round_float(1.01),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int(count),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```