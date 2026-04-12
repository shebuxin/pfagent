```python
# required_dependencies: andes
import andes
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Identify the buses and their voltages
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v

# Find the indices of the three lowest-voltage buses
lowest_voltage_indices = np.argsort(voltages)[:3]
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[i], 6) for i in lowest_voltage_indices]

# Construct the result JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print("RESULT_JSON=",RESULT_JSON)
```