```python
# required_dependencies: numpy,json,matplotlib
import numpy as np
import json
import andes
import matplotlib.pyplot as plt

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 9,
        "idx": "PQ_VERIFY_060_A",
        "p0": 0.019,
        "q0": 0.012,
    },
)

ssa.setup()
scale_factor = 1.06
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
plot_file = "scenario_060_turn3_line.png"
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
    "scale_factor": _round_float(1.06),
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
    "plot_file": plot_file,
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```