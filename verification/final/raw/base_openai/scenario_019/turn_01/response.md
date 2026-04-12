```python
# required_dependencies: numpy,pandas,andes
import os
import numpy as np
import pandas as pd
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_019.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count buses above threshold
threshold = 1.015
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
voltages = ssa.Bus.v.v
lowest_indices = np.argsort(voltages)[:2]
lowest_bus_ids = ssa.Bus.idx.v[lowest_indices].tolist()
lowest_voltages = np.round(voltages[lowest_indices], 6).tolist()

# Prepare the result JSON
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print("RESULT_JSON=", RESULT_JSON)
```