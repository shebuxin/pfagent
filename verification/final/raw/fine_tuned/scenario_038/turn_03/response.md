```python
# required_dependencies: andes,json,numpy,matplotlib
import andes
import json
import numpy as np
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
pv_setpoint = 1.01
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])
scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
plot_file = "scenario_038_turn3_bar.png"
plt.figure(figsize=(10, 4))
plt.bar(bus_ids.astype(str), bus_v)
plt.xticks(rotation=90)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
min_index = int(int(np.argmin(bus_v)))
max_index = int(int(np.argmax(bus_v)))
result = {
    "scale_factor": _round_float(1.03),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```