```python
# required_dependencies: numpy,json
import os
import numpy as np
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_023.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v
bus_numbers = ssa.Bus.idx.v

# Determine max and min voltages and their corresponding buses
max_voltage = float(np.max(bus_voltages))
min_voltage = float(np.min(bus_voltages))
max_bus = int(bus_numbers[np.argmax(bus_voltages)])
min_bus = int(bus_numbers[np.argmin(bus_voltages)])

# Prepare the RESULT_JSON output
result_json = {
    "max_bus": max_bus,
    "max_voltage": round(max_voltage, 6),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6)
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')
```