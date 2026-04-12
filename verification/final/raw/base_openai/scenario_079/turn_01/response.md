Here's a complete runnable Python script that loads your Excel file, runs a power flow analysis, and reports the maximum and minimum voltage buses as specified:

```python
# required_dependencies: andes
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_079.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Run power flow analysis
andes.run_power_flow(ssa)

# Inspect voltage values
bus_voltages = ssa.V.bus.v
bus_indices = ssa.V.idx.v

# Find maximum and minimum voltage buses
max_voltage_idx = bus_indices[bus_voltages.argmax()]
min_voltage_idx = bus_indices[bus_voltages.argmin()]
max_voltage = round(bus_voltages.max(), 6)
min_voltage = round(bus_voltages.min(), 6)

# Prepare the JSON result
result_json = {
    "max_bus": int(max_voltage_idx),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_voltage_idx),
    "min_voltage": float(min_voltage),
}

# Print the result as specified
print(f"RESULT_JSON={json.dumps(result_json)}")
``` 

This script runs through the necessary steps to load your case, execute the power flow analysis, and extract the results you want in the requested format.