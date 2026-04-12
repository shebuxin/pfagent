```python
# required_dependencies: andes,numpy,matplotlib
import json
import andes
import numpy as np
import os
import matplotlib.pyplot as plt

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)

pv_setpoint = 1.01
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_pos = int(np.where(bus_ids == pv_bus)[0][0])
pv_voltage = round(float(bus_v[pv_pos]), 6)

plot_file = "scenario_009_turn3_bar.png"
plt.figure(figsize=(10, 4))
plt.bar(bus_ids, bus_v, width=0.8)
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Bus Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

result_json = {}
result_json["scale_factor"] = round(float(scale_factor), 6)
result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])
result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)
result_json["plot_file"] = plot_file

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```