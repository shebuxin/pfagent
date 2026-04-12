```python
# required_dependencies: numpy, pandas, andes
import numpy as np
import pandas as pd
import andes

# Load the built-in IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(case)

# Extract bus voltage data and find the two lowest voltage buses
voltage_data = case.Bus.v
bus_ids = case.Bus.idx

# Get indices of the two lowest voltages
lowest_voltage_indices = np.argsort(voltage_data)[:2]

# Prepare the result JSON
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = [round(voltage_data[idx], 6) for idx in lowest_voltage_indices]

# Print the result JSON
RESULT_JSON = {'selected_bus_ids': selected_bus_ids, 'selected_voltages': selected_voltages}
print("RESULT_JSON=", RESULT_JSON)
```