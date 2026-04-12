```python
# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_053.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Inspect results to find max and min voltage buses
voltages = ssa.V.v
buses = ssa.bus.v

max_idx = np.argmax(voltages)
min_idx = np.argmin(voltages)

max_bus = int(buses[max_idx])
max_voltage = round(float(voltages[max_idx]), 6)
min_bus = int(buses[min_idx])
min_voltage = round(float(voltages[min_idx]), 6)

# Create the result JSON
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```