```python
# required_dependencies: andes
import os
import andes

# Load the case
case = os.path.join(os.getcwd(), "verify_ieee14_024.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa, solver='default', log=True)

# Analyze voltage results
voltages = ssa.bus.v
max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

# Prepare results
max_bus = ssa.bus.idx.v[max_voltage_idx]
max_voltage = round(voltages[max_voltage_idx], 6)
min_bus = ssa.bus.idx.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Print result in specified JSON format
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}
print(f"RESULT_JSON={RESULT_JSON}")
```