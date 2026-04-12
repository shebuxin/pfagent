# required_dependencies: pandas
import os
import andes
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_022.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.040
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v]
slack_setpoint = 1.040
ssa.Bus.v.set(slack_bus_idx, slack_setpoint)

# Add a new PQ load at bus 10
new_load_idx = 'PQ_VERIFY_022_B'
p0 = 0.018
q0 = 0.013
bus_number = 10

# Register new PQ load
ssa.PQ.add(bus=bus_number, idx=new_load_idx, p0=p0, q0=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Get max and min voltage buses
max_bus_idx = ssa.Bus.idx.v[np.argmax(ssa.Bus.v.v)]
min_bus_idx = ssa.Bus.idx.v[np.argmin(ssa.Bus.v.v)]
max_voltage = round(ssa.Bus.v.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v.v[min_bus_idx], 6)

# Count total PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus_idx),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus_idx),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(RESULT_JSON))