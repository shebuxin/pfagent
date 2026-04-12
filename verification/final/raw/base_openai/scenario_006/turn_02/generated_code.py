# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True)

# Set the slack-bus voltage target
slack_setpoint = 1.025
case.Slack.set("Vtarget", slack_setpoint)

# Rerun power flow
case.run()

# Get the slack bus voltage
slack_bus = case.Slack.idx.v[0]  # Assuming there's only one slack bus
slack_voltage = round(case.Bus.v[slack_bus], 6)

# Count buses below 1.005 p.u.
selected_count = (case.Bus.v < 1.005).sum()

# Prepare the result in required JSON format
RESULT_JSON = json.dumps({
    "slack_bus": slack_bus,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
})

print(f"RESULT_JSON={RESULT_JSON}")