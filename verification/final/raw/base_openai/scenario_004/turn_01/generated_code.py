# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.fpn(ssa)

# Get the slack bus voltage
slack_bus = ssa.Slack.bus.v[0]
slack_voltage = round(ssa.Slack.V.v[0], 6)

# Get all bus voltages and their IDs
bus_voltages = ssa.Bus.V.v
bus_ids = ssa.Bus.idx.v

# Sort buses by voltage to get the top 5 highest
top_indices = bus_voltages.argsort()[-5:][::-1]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = [round(bus_voltages[i], 6) for i in top_indices]

# Prepare the RESULT_JSON
RESULT_JSON = {
    'slack_bus': int(slack_bus),
    'slack_voltage': slack_voltage,
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages
}

# Print the result in the specified format
print(f"RESULT_JSON={RESULT_JSON}")