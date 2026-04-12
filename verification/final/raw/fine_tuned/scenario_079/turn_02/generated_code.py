# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case_path = os.path.join(cwd, "verify_kundur_079.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[0.99])
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
pv_bus = int(ssa.PV.bus.v[0])
pv_v = float(ssa.Bus.v.v[bus_ids.tolist().index(pv_bus)])
threshold = 0.99
mask = bus_v > threshold
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(0.99),
    "pv_voltage": _round_float(pv_v),
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))