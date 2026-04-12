# required_dependencies: pandas, andes
import pandas as pd
import andes

# Load the case
case = 'verify_kundur_078.xlsx'
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
slack_setpoint = 1.020
ssa.Bus.v_set.v[slack_bus_idx] = slack_setpoint

# Rerun power flow
andes.run_power_flow(ssa)

# Get updated slack bus voltage
slack_voltage = ssa.Bus.v.v[slack_bus_idx]

# Count how many buses fall below 0.970 p.u.
selected_count = sum(1 for voltage in ssa.Bus.v.v if voltage < 0.970)

# Prepare the result JSON-like structure
RESULT_JSON = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")