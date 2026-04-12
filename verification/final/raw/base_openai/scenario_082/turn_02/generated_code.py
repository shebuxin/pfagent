# required_dependencies: pandas
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_082.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
bus_idx = 4
load_id = 'PQ_VERIFY_082_D'
p0 = 0.014
q0 = 0.008

# Set the new PQ load
ss_load_idx = ssa.PQ.add(bus=bus_idx, idx=load_id, p0=p0, q0=q0)

# Rerun power flow
andes.run_powerflow()

# Get the slack bus information
slack_bus = ssa.Slack.bus.v[0]
slack_voltage = round(float(ssa.Bus.voltage.v[slack_bus]), 6)

# Find buses below 0.940 p.u. voltage
threshold = 0.940
selected_buses = ssa.Bus.voltage.v < threshold
selected_bus_ids = [int(ssa.Bus.id.v[i]) for i in range(len(selected_buses)) if selected_buses[i]]
selected_count = len(selected_bus_ids)

# Create the result JSON
RESULT_JSON = {
    "added_load_idx": load_id,
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")