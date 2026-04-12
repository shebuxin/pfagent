import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=True,
    no_output=True,
)

ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[1.015])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_index = int(np.where(bus_ids == pv_bus)[0][0])
threshold = 1.04
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(1.015),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int(np.sum(bus_v > threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))