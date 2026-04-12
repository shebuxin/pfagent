# required_dependencies: pandas
import pandas as pd
import andes

# Load the built-in Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v[0]  # Assuming there's only one slack bus
slack_setpoint = 1.010
ssa.Slack.set_voltage_target(slack_bus_idx, slack_setpoint)

# Rerun power flow
andes.run_power_flow(ssa)

# Retrieve values
slack_voltage = round(ssa.Slack.voltage.v[0], 6)

# Count buses below 0.960 p.u.
below_threshold = ssa.Bus.voltage.v < 0.960
selected_count = sum(below_threshold)

# Constructing the final JSON-like output
RESULT_JSON = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")