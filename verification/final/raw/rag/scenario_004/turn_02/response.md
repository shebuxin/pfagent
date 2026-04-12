```python
# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 10,
        "idx": "PQ_VERIFY_004_A",
        "p0": 0.019,
        "q0": 0.012,
    },
)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

result_json = {}
threshold = 1.015
below_mask = bus_v < threshold
result_json["added_load_idx"] = "PQ_VERIFY_004_A"
result_json["added_load_bus"] = 10
result_json["threshold"] = round(float(threshold), 6)
result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]
result_json["selected_count"] = int(np.sum(below_mask))
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```