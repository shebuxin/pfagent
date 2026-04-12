# required_dependencies: andes,matplotlib,json
import andes
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_pjm5_093.json")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 1,
        "idx": "PQ_VERIFY_093_E",
        "p0": 0.012,
        "q0": 0.006,
    },
)

ssa.setup()
scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
plot_file = "scenario_093_turn3_voltage.png"
plt.figure(figsize=(10, 4))
plt.plot(bus_ids, bus_v, marker="o", linewidth=1.5)
plt.xticks(rotation=90)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
max_index = int(np.argmax(bus_v))
min_index = int(np.argmin(bus_v))
result = {
    "added_load_idx": "PQ_VERIFY_093_E",
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))