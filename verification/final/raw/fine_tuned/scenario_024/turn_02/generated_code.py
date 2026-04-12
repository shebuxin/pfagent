# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_ieee14_024.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.015])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
pv_bus = int(ssa.PV.bus.v[0])
pv_index = int(bus_ids.tolist().index(pv_bus))
threshold = 1.02
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(1.015),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int((bus_v > threshold).sum()),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))