# required_dependencies: andes
import andes

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract bus voltages and their corresponding IDs
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v

# Combine bus ids and their voltages into a list of tuples
bus_voltage_pairs = list(zip(bus_ids, voltages))

# Sort by voltage
sorted_buses = sorted(bus_voltage_pairs, key=lambda x: x[1])

# Select the 4 lowest-voltage buses
lowest_buses = sorted_buses[:4]

# Separate the bus IDs and voltages
selected_bus_ids = [int(bus[0]) for bus in lowest_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_buses]

# Print the result in the required format
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print(f"RESULT_JSON={RESULT_JSON}")