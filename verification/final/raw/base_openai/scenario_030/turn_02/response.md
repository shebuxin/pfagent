```python
# required_dependencies: pandas
import pandas as pd
import andes
import numpy as np
import os

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
bus_idx = np.where(ssa.Bus.idx.v == 10)[0][0]
ssa.PQ.add(idx='PQ_VERIFY_030_A', bus=bus_idx, p0=0.013, q0=0.008)

# Rerun power flow
andes.run_power_flow(ssa)

# Get bus voltages
voltages = ssa.Bus.voltage.v
bus_ids = ssa.Bus.idx.v

# Define threshold for minimum voltage
threshold = 0.960

# Identify buses below threshold and find the minimum voltage bus
selected_bus_indices = np.where(voltages < threshold)[0]
selected_bus_ids = list(bus_ids[selected_bus_indices])
selected_count = len(selected_bus_ids)
min_voltage_idx = np.argmin(voltages)
min_bus = int(bus_ids[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": "PQ_VERIFY_030_A",
    "added_load_bus": 10,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```