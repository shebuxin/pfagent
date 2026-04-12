# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get the slack bus and its voltage
slack_bus_idx = ssa.BUS.idx.v[ssa.BUS.slack.v]
slack_voltage = ssa.BUS.v[slack_bus_idx]

# Get bus voltages and their indices
bus_voltages = ssa.BUS.v
bus_ids = ssa.BUS.idx.v

# Pair bus voltages with their indices and sort by voltage
bus_voltage_pairs = list(zip(bus_ids, bus_voltages))
top_buses = sorted(bus_voltage_pairs, key=lambda x: x[1], reverse=True)[:4]

# Prepare RESULT_JSON
selected_bus_ids = [int(bus[0]) for bus in top_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in top_buses]

RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")