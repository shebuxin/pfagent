Here's the complete Python script that incorporates your latest requirements. It adds a new PQ load at bus 9, keeps the slack-bus setting, reruns the power flow, and reports the required information in the specified JSON format.

```python
# required_dependencies: numpy,pandas,json
import os
import andes
import numpy as np
import pandas as pd
import json

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[0]  # Assuming the first bus is the slack bus
slack_setpoint = 1.020
ssa.Bus.v.set(slack_setpoint, idx=slack_bus_idx)

# Add a new PQ load at bus 9
new_load_idx = ssa.PQ.add("PQ_VERIFY_064_B", bus=9, p0=0.018, q0=0.013)

# Run the power flow
andes.run_power_flow(ssa)

# Get the maximum and minimum voltage buses
max_voltage_idx = np.argmax(ssa.Bus.v.v)
min_voltage_idx = np.argmin(ssa.Bus.v.v)

# Collect required results
max_bus = max_voltage_idx
max_voltage = round(ssa.Bus.v.v[max_bus], 6)
min_bus = min_voltage_idx
min_voltage = round(ssa.Bus.v.v[min_bus], 6)
total_pq_count = len(ssa.PQ.idx.v)

# Create the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')
```