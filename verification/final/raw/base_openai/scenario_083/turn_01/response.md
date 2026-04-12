```python
# required_dependencies: andes
import andes
import os
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_083.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Identify the buses and their voltages
voltages = ssa.Bus.V.v
bus_ids = ssa.Bus.idx.v

# Find the 3 lowest-voltage buses
lowest_indices = np.argsort(voltages)[:3]
selected_bus_ids = bus_ids[lowest_indices].tolist()
selected_voltages = np.round(voltages[lowest_indices].tolist(), 6).tolist()

# Prepare result in JSON format
result_json = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print(f"RESULT_JSON={json.dumps(result_json)}")
```