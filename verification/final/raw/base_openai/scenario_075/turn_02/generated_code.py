# required_dependencies: pandas
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_075.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the slack bus ID
slack_setpoint = 0.990
ssa.Bus.vmag.set(slack_setpoint, index=ssa.Bus.idx.v == slack_bus)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage and count buses below 0.940 p.u.
slack_voltage = ssa.Bus.vmag.v[ssa.Bus.idx.v == slack_bus].item()
selected_count = (ssa.Bus.vmag.v < 0.940).sum()

# Prepare result in JSON format
result_json = {
    "slack_bus": int(slack_bus),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": int(selected_count)
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")