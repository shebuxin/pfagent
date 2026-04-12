```python
# required_dependencies: andes,numpy,matplotlib
import json
import andes
import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_096.json")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 4,
        "idx": "PQ_VERIFY_096_E",
        "p0": 0.018,
        "q0": 0.012,
    },
)

ssa.setup()

scale_factor = 1.06
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

plot_file = "scenario_096_turn3_voltage.png"
plt.figure(figsize=(10, 4))
plt.plot(bus_ids, bus_v, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Bus Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

result_json = {}
result_json["added_load_idx"] = "PQ_VERIFY_096_E"
result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])
result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)
result_json["plot_file"] = plot_file

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```