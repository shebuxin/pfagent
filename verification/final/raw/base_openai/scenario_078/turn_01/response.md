```python
# required_dependencies: pandas, andes
import pandas as pd
import andes

# Load the case
case = 'verify_kundur_078.xlsx'
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get bus voltage data
voltages = ssa.Bus.v.v

# Define threshold
threshold = 1.020

# Find selected buses
selected_bus_ids = [idx for idx, voltage in enumerate(voltages) if voltage > threshold]
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_buses = sorted(range(len(voltages)), key=lambda x: voltages[x])[:2]
lowest_bus_ids = [lowest_buses[0], lowest_buses[1]]
lowest_voltages = [round(voltages[lowest_buses[0]], 6), round(voltages[lowest_buses[1]], 6)]

# Prepare the result JSON-like structure
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```