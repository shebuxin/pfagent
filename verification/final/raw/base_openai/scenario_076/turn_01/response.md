```python
# required_dependencies: pandas, numpy
import os
import json
import andes
import numpy as np

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_076.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Set the voltage threshold
threshold = 1.0

# Count all buses above the threshold
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_buses_indices = np.argsort(ssa.Bus.v)[:2]
lowest_bus_ids = ssa.Bus.idx.v[lowest_buses_indices].tolist()
lowest_voltages = [round(ssa.Bus.v[i], 6) for i in lowest_buses_indices]

# Prepare the result as JSON
result_json = {
    "threshold": float(threshold),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")
```