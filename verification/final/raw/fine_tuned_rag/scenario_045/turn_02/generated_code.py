# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_045.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 15,
        "idx": "PQ_VERIFY_045_A",
        "p0": 0.016,
        "q0": 0.01,
    },
)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

result_json = {}
threshold = 0.97
below_mask = bus_v < threshold
result_json["added_load_idx"] = "PQ_VERIFY_045_A"
result_json["added_load_bus"] = 15
result_json["threshold"] = round(float(threshold), 6)
result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]
result_json["selected_count"] = int(np.sum(below_mask))
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))