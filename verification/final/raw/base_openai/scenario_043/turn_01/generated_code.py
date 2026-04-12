# required_dependencies: pandas
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_043.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Get slack bus information
slack_bus_idx = ssa.Bus.slack.idx[0]
slack_voltage = ssa.Bus.v[slack_bus_idx]

# Get bus voltages and indices
bus_voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Get top 2 highest voltage buses
top_indices = np.argsort(bus_voltages)[-2:][::-1]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = bus_voltages[top_indices].tolist()

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": [round(float(v), 6) for v in selected_voltages]
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")