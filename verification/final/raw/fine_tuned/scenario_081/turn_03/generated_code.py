# required_dependencies: andes,json,matplotlib
import andes
import json
import os
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.01])
scale_factor = 1.05
ssa.PQ.set(src="v0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.v0.v)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
plot_file = "scenario_081_turn3_bar.png"
plt.figure(figsize=(10, 4))
plt.bar(bus_ids, bus_v)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
min_index = int(min(range(len(bus_v)), key=bus_v.__getitem__))
max_index = int(max(range(len(bus_v)), key=bus_v.__getitem__))
result = {
    "scale_factor": _round_float(1.05),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))