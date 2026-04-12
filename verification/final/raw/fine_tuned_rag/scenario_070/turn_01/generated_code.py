# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

result_json = {}
top_k = 4
selected_idx = np.argsort(bus_v)[:top_k]
result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]
result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))