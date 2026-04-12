```python
# required_dependencies: pandas
import os
import andes
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_028.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get voltages and bus IDs
voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Find the 4 lowest-voltage buses
lowest_voltages_indices = voltages.argsort()[:4]
selected_bus_ids = bus_ids[lowest_voltages_indices].tolist()
selected_voltages = [round(voltages[i], 6) for i in lowest_voltages_indices]

# Prepare the result JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")
```