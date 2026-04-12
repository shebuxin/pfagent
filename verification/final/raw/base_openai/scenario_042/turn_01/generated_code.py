# required_dependencies: andes
import andes

# Load the built-in IEEE 39 case
case_path = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Extract bus voltages
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.Vmag.v

# Combine bus IDs and voltages, then find the 4 lowest-voltage buses
voltages_buses = sorted(zip(bus_ids, voltages), key=lambda x: x[1])[:4]
selected_bus_ids = [int(bus_id) for bus_id, _ in voltages_buses]
selected_voltages = [round(voltage, 6) for _, voltage in voltages_buses]

# Print the result in the specified format
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print(f"RESULT_JSON={RESULT_JSON}")