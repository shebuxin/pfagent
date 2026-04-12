# required_dependencies: andes,numpy,json,matplotlib
import andes
import numpy as np
import json
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

case = "verify_ieee39_051.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.005])
scale_factor = 1.02
ssa.PQ.set(src="v0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.v0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
plot_file = "scenario_051_turn3_bar.png"
plt.figure(figsize=(10, 4))
plt.bar(bus_ids.astype(str), bus_v)
plt.xticks(rotation=90)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
min_index = int(np.argmin(bus_v))
max_index = int(np.argmax(bus_v))
result = {
    "scale_factor": _round_float(1.02),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))