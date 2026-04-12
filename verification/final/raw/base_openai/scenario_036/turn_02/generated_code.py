# required_dependencies: andes
import andes
import numpy as np
import json

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_index = ssa.bus.slack.v[0]  # Getting the index of the slack bus
ssa.PV.set(slack_bus_index, v_set=1.035)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage and count buses below the threshold
slack_voltage = round(ssa.Bus.v[slack_bus_index], 6)
slack_setpoint = 1.035
selected_count = np.sum(ssa.Bus.v < 0.980)

# Prepare the JSON result
result_json = {
    "slack_bus": int(slack_bus_index),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

# Print the result in the specified format
print(f"RESULT_JSON={json.dumps(result_json)}")