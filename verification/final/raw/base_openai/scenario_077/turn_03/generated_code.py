# required_dependencies: pandas
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_077.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.type.v == 3][0]  # Assuming type 3 is the slack bus
slack_setpoint = 1.010
ssa.Bus.v.target.v[slack_bus] = slack_setpoint

# Add a new PQ load at bus 7
new_pq_idx = 'PQ_VERIFY_077_B'
p0 = 0.016
q0 = 0.011
bus_number = 7
andes.create_PQ(ssa, new_pq_idx, bus_number, p0, q0)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get the maximum and minimum voltage buses
max_bus = ssa.Bus.idx.v[ssa.Bus.v.v.argmax()]
max_voltage = round(ssa.Bus.v.v.max(), 6)
min_bus = ssa.Bus.idx.v[ssa.Bus.v.v.argmin()]
min_voltage = round(ssa.Bus.v.v.min(), 6)
total_pq_count = ssa.PQ.idx.v.size

# Form the result
RESULT_JSON = {
    "added_load_idx": new_pq_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": int(total_pq_count)
}

print(f"RESULT_JSON={RESULT_JSON}")