```python
# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case from the uploaded Excel file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Retrieve bus voltages and their corresponding bus indices
voltages = ssa.bus.v
bus_indices = np.arange(len(voltages))

# Find max and min voltage buses
max_index = np.argmax(voltages)
min_index = np.argmin(voltages)

max_bus = int(ssa.bus.idx.v[max_index])
max_voltage = round(float(voltages[max_index]), 6)
min_bus = int(ssa.bus.idx.v[min_index])
min_voltage = round(float(voltages[min_index]), 6)

# Prepare the RESULT_JSON
result_json = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result in the specified format
print(f"RESULT_JSON={result_json}")
```