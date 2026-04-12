# required_dependencies: pandas
import pandas as pd
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get bus voltages
voltages = ssa.Bus.voltage.v
bus_ids = ssa.Bus.idx.v

# Identify the slack bus and its voltage
slack_bus_idx = ssa.Bus.slack.v[0]
slack_voltage = float(voltages[slack_bus_idx])

# Get top 3 highest voltage buses
high_voltage_indices = voltages.argsort()[-3:][::-1]
selected_bus_ids = list(bus_ids[high_voltage_indices])
selected_voltages = [round(float(voltages[idx]), 6) for idx in high_voltage_indices]

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)