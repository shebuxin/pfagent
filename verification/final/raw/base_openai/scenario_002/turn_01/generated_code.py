# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus and its voltage
slack_bus_idx = ssa.slack.v[0]
slack_voltage = ssa.Bus.v[slack_bus_idx]

# Get bus voltages and their indices
voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx.v

# Get top 3 highest-voltage buses
top_indices = sorted(range(len(voltages)), key=lambda i: voltages[i], reverse=True)[:3]
selected_bus_ids = [bus_ids[i] for i in top_indices]
selected_voltages = [round(voltages[i], 6) for i in top_indices]

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")