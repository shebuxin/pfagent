# required_dependencies: numpy,andes
import os
import numpy as np
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_046.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.power_flow(ssa)

# Get slack bus voltage
slack_bus_idx = ssa.Slack.idx.v[0]
slack_bus_voltage = ssa.Bus.v[slack_bus_idx]

# Get all bus voltages and their indices
bus_voltages = ssa.Bus.v
bus_ids = np.arange(len(bus_voltages))

# Get top 5 highest voltage buses
top_indices = np.argsort(-bus_voltages)[:5]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = np.round(bus_voltages[top_indices].tolist(), 6)

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_bus_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")