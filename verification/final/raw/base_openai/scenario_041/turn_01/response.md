```python
# required_dependencies: pandas
import andes
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Identify the three lowest-voltage buses
voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx.v
lowest_voltages_idx = np.argsort(voltages)[:3]

# Prepare the result
selected_bus_ids = bus_ids[lowest_voltages_idx].tolist()
selected_voltages = np.round(voltages[lowest_voltages_idx].tolist(), 6).tolist()

# Print the result in the specified JSON format
RESULT_JSON = {"selected_bus_ids": selected_bus_ids, "selected_voltages": selected_voltages}
print("RESULT_JSON=", RESULT_JSON)
```