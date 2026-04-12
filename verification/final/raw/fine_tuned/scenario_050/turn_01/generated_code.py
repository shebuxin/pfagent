# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
threshold = 1.05
mask = bus_v > threshold
low_rank = sorted(zip(bus_ids, bus_v))[:2]
result = {
    "threshold": _round_float(threshold),
    "selected_bus_ids": [int(value) for value in bus_ids[mask]],
    "selected_count": int(mask.sum()),
    "lowest_bus_ids": [int(key) for key, _ in low_rank],
    "lowest_voltages": [_round_float(value) for _, value in low_rank],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))