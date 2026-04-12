# required_dependencies: andes,json
import andes
import json
import os

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
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
threshold = 0.96
mask = bus_v < threshold
min_index = int(bus_ids[mask].tolist()[bus_v[mask].argmin()])
result = {
    "added_load_idx": "PQ_VERIFY_044_A",
    "added_load_bus": 10,
    "threshold": _round_float(threshold),
    "selected_bus_ids": sorted([int(value) for value in bus_ids[mask].tolist()]),
    "selected_count": int(mask.sum()),
    "min_bus": min_index,
    "min_voltage": _round_float(bus_v[bus_ids.tolist().index(min_index)]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))