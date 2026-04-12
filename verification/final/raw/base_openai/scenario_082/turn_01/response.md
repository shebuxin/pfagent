```python
# required_dependencies: pandas
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_082.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow()

# Get voltage information
voltages = ssa.Bus.voltage.v
bus_ids = ssa.Bus.id.v

# Find the indices of the two lowest voltage buses
lowest_volt_indices = voltages.argsort()[:2]

# Prepare the result
selected_bus_ids = [int(bus_ids[idx]) for idx in lowest_volt_indices]
selected_voltages = [round(float(voltages[idx]), 6) for idx in lowest_volt_indices]

# Create the result JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```