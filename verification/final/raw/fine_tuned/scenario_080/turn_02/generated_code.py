# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

case = os.path.join(os.getcwd(), "verify_kundur_080.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=ssa.PV.idx.v[0], attr="v", value=1.0)
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
pv_bus = ssa.PV.bus.v[0]
pv_index = int(bus_ids.index(pv_bus))
threshold = 1.0
Count = sum(v > threshold for v in bus_v)
result_json = json.dumps(
    {
        "pv_bus": int(pv_bus),
        "pv_setpoint": _round_float(1.0),
        "pv_voltage": _round_float(bus_v[pv_index]),
        "selected_count": int(Count),
    },
    sort_keys=True,
)
print("RESULT_JSON=" + result_json)