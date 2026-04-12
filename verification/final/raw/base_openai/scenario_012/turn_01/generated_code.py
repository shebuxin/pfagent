# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Run the power flow
case.run()

# Get voltage data
voltages = case.Bus.v.v  # Bus voltages
bus_ids = case.Bus.idx.v  # Bus IDs

# Find the 2 lowest-voltage buses
lowest_voltage_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])[:2]
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_indices]
selected_voltages = [round(float(voltages[i]), 6) for i in lowest_voltage_indices]

# Create the result JSON object
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")