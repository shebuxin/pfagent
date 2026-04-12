# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_075.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

result_json = {}
threshold = 0.99
above_mask = bus_v > threshold
lowest_idx = np.argsort(bus_v)[:2]
result_json["threshold"] = round(float(threshold), 6)
result_json["selected_bus_ids"] = [int(item) for item in bus_ids[above_mask]]
result_json["selected_count"] = int(np.sum(above_mask))
result_json["lowest_bus_ids"] = [int(bus_ids[i]) for i in lowest_idx]
result_json["lowest_voltages"] = [round(float(bus_v[i]), 6) for i in lowest_idx]

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))