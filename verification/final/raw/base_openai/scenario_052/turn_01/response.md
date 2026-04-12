```python
# required_dependencies: numpy,pandas,andes
import numpy as np
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_052.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
result = andes.run_power_flow(ssa)

# Extract bus voltages
voltages = ssa.Bus.v.v

# Find maximum and minimum voltage buses
max_idx = np.argmax(voltages)
min_idx = np.argmin(voltages)

max_bus = ssa.Bus.idx.v[max_idx]
max_voltage = round(float(voltages[max_idx]), 6)
min_bus = ssa.Bus.idx.v[min_idx]
min_voltage = round(float(voltages[min_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")
```