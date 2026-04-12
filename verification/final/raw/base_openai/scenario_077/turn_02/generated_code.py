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

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.v.v[slack_bus], 6)

# Count buses below 0.960 p.u.
selected_count = (ssa.Bus.v.v < 0.960).sum()

# Form the result
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")