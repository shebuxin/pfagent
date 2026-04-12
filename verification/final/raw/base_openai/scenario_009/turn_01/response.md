```python
# required_dependencies: andes
import andes
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Inspect bus voltages
bus_voltages = ssa.Bus.Vm.v
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)

# Get corresponding bus indices
max_bus = int(np.where(bus_voltages == max_voltage)[0][0]) + 1  # Convert to 1-based index
min_bus = int(np.where(bus_voltages == min_voltage)[0][0]) + 1  # Convert to 1-based index

# Prepare the JSON result
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": min_bus,
    "min_voltage": round(float(min_voltage), 6)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```