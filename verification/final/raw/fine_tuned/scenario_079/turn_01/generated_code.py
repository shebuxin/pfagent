# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_kundur_079.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
max_index = int(np.argmax(bus_v))
min_index = int(np.argmin(bus_v))
result = {
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))