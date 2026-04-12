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

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage
slack_voltage = round(ssa.Bus.v.v[slack_bus_idx][0], 6)

# Count buses below the threshold of 1.015 p.u.
selected_count = np.sum(ssa.Bus.v.v < 1.015)

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx[0]),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(RESULT_JSON))