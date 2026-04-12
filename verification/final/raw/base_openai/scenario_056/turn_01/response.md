```python
# required_dependencies: andes
import andes
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_056.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Retrieve bus voltages and IDs
voltage_data = ssa.Bus.voltage.v
bus_ids = ssa.Bus.idx.v

# Combine bus IDs and their voltages, and sort by voltage
bus_voltages = list(zip(bus_ids, voltage_data))
sorted_buses = sorted(bus_voltages, key=lambda x: x[1])

# Get the 4 lowest-voltage buses
lowest_buses = sorted_buses[:4]

# Prepare the result
selected_bus_ids = [bus[0] for bus in lowest_buses]
selected_voltages = [round(bus[1], 6) for bus in lowest_buses]

# Print the result in the required format
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print(f"RESULT_JSON={RESULT_JSON}")
```