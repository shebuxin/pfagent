# required_dependencies: pandas, numpy
import os
import json
import andes
import numpy as np

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_076.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Slack.idx.v[0]  # Assuming there is one Slack bus
slack_setpoint = 1.0
ssa.Slack.vset.v[0] = slack_setpoint

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.v[ssa.Bus.idx.v == slack_bus][0], 6)

# Count how many buses fall below 0.950 p.u.
selected_count = np.sum(ssa.Bus.v < 0.950)

# Prepare the result as JSON
result_json = {
    "slack_bus": int(slack_bus),
    "slack_setpoint": float(slack_setpoint),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")