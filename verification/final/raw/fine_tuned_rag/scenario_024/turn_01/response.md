```python
# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_024.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)

result_json = {}
result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])
result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```