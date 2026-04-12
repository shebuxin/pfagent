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

# Run power flow
andes.run_powerflow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.v.v[np.where(ssa.Bus.idx.v == slack_bus_id)[0][0]], 6)

# Count how many buses fall below 0.950 p.u.
selected_count = np.sum(ssa.Bus.v.v < 0.950)

# Prepare the result JSON
result_json = {
    "slack_bus": slack_bus_id,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_count": selected_count,
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(result_json))