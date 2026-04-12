Here’s the complete runnable Python script that meets your requirements:

```python
# required_dependencies: andes
import andes

# Load the case
case = "verify_ieee39_051.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Find maximum and minimum voltage buses
voltages = ssa.Bus.v
max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

max_bus = ssa.Bus.idx.v[max_voltage_idx]
max_voltage = round(voltages[max_voltage_idx], 6)
min_bus = ssa.Bus.idx.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Prepare the result
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
}

# Print the result in the required format
print(f'RESULT_JSON={RESULT_JSON}')
``` 

This script will load your uploaded Excel file, run the power flow analysis, and then find and print the buses with the maximum and minimum voltages as specified.