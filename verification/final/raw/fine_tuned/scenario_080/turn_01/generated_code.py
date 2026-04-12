# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_kundur_080.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
max_index = int(list(bus_v).index(max(bus_v)))
min_index = int(list(bus_v).index(min(bus_v)))
result_json = json.dumps(
    {
        "max_bus": int(bus_ids[max_index]),
        "max_voltage": _round_float(bus_v[max_index]),
        "min_bus": int(bus_ids[min_index]),
        "min_voltage": _round_float(bus_v[min_index]),
    },
    sort_keys=True,
)
print("RESULT_JSON=" + result_json)