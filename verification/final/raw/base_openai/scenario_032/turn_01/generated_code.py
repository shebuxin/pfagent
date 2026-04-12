# required_dependencies: andes
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Extracting slack bus voltage and identifying top 5 highest voltage buses
slack_bus = ssa.bus.idx[ssa.bus.slack.v][0]
slack_voltage = round(ssa.bus.Vmag.v[slack_bus], 6)
voltages = ssa.bus.Vmag.v
bus_ids = ssa.bus.idx.v

# Getting top 5 highest voltage buses
sorted_indices = sorted(range(len(voltages)), key=lambda i: voltages[i], reverse=True)[:5]
selected_bus_ids = [int(bus_ids[i]) for i in sorted_indices]
selected_voltages = [round(float(voltages[i]), 6) for i in sorted_indices]

# Prepare Result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print("RESULT_JSON=", RESULT_JSON)