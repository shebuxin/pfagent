```python
# required_dependencies: andes,json,matplotlib
import andes
import json
import os
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_ieee39_044.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 10,
        "idx": "PQ_VERIFY_044_A",
        "p0": 0.013,
        "q0": 0.008,
    },
)

ssa.setup()
scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
plot_file = "scenario_044_turn3_line.png"
plt.figure(figsize=(10, 4))
plt.plot(bus_ids, bus_v, marker="o", linewidth=1.5)
plt.xticks(rotation=90)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()
max_index = int(bus_ids.tolist()[bus_v.argmax()])
min_index = int(bus_ids.tolist()[bus_v.argmin()])
result = {
    "scale_factor": _round_float(1.03),
    "max_bus": max_index,
    "max_voltage": _round_float(bus_v.max()),
    "min_bus": min_index,
    "min_voltage": _round_float(bus_v.min()),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```