# required_dependencies: numpy,pandas,andes
import os
import numpy as np
import pandas as pd
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_019.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the slack bus index
ssa.Bus.v.set(slack_bus_idx, 1.020)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_019_B'
ssa.PQ.add(idx=new_load_idx, bus=4, p0=0.012, q0=0.007)

# Rerun power flow
andes.run_power_flow(ssa)

# Determine max-voltage and min-voltage bus
max_voltage = round(np.max(ssa.Bus.v.v), 6)
min_voltage = round(np.min(ssa.Bus.v.v), 6)
max_bus = ssa.Bus.idx.v[np.argmax(ssa.Bus.v.v)]
min_bus = ssa.Bus.idx.v[np.argmin(ssa.Bus.v.v)]

# Count the total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print("RESULT_JSON=", RESULT_JSON)