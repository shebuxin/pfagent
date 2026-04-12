```python
# required_dependencies: andes,json
import andes
import json
import numpy as np

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
pv_setpoint = 1.01
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_index = int(np.where(bus_ids == pv_bus)[0][0])
threshold = 1.01
count = int(np.sum(bus_v > threshold))
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(1.01),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": count,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```