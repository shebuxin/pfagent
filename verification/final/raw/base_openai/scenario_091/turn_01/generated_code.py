# required_dependencies: andes,json
import andes
import json

# Load PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Extract voltage information
voltages = ssa.Bus.v
slack_bus_idx = ssa.Slack.idx.v
slack_bus = ssa.Slack.bus.v[0]
slack_voltage = round(voltages[slack_bus_idx][0], 6)

max_bus_idx = voltages.argmax()
max_bus = ssa.Bus.idx.v[max_bus_idx]
max_voltage = round(voltages[max_bus_idx], 6)

min_bus_idx = voltages.argmin()
min_bus = ssa.Bus.idx.v[min_bus_idx]
min_voltage = round(voltages[min_bus_idx], 6)

# Prepare JSON output
result_json = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")