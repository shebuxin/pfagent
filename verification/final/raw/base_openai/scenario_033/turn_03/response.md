```python
# required_dependencies: numpy,json
import andes
import numpy as np
import json
import os

# Load the built-in IEEE 39 case
case_path = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_id = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
slack_setpoint = 1.015
ssa.Bus.v.set(slack_setpoint, indices=np.where(ssa.Bus.idx.v == slack_bus_id)[0])

# Add a new PQ load at bus 4
new_load_idx = "PQ_VERIFY_033_B"
p0 = 0.012
q0 = 0.007
bus_4_idx = np.where(ssa.Bus.idx.v == 4)[0]
if bus_4_idx.size == 0:
    raise ValueError("Bus 4 does not exist.")
ssa.PQ.add(idx=new_load_idx, bus=4, p0=p0, q0=q0)

# Run power flow
andes.run_powerflow(ssa)

# Identify the maximum and minimum voltage buses
max_voltage_bus_idx = np.argmax(ssa.Bus.v.v)
min_voltage_bus_idx = np.argmin(ssa.Bus.v.v)
max_bus = ssa.Bus.idx.v[max_voltage_bus_idx]
min_bus = ssa.Bus.idx.v[min_voltage_bus_idx]
max_voltage = round(ssa.Bus.v.v[max_voltage_bus_idx], 6)
min_voltage = round(ssa.Bus.v.v[min_voltage_bus_idx], 6)

# Count the total number of PQ loads now present
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count,
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(result_json))
```