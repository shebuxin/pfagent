# required_dependencies: andes
import andes

# Load the Kundur full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Retrieve voltage data
voltage_data = ssa.bus.voltage.v
bus_ids = ssa.bus.idx.v

# Find the three lowest voltage buses
lowest_voltage_indices = sorted(range(len(voltage_data)), key=lambda i: voltage_data[i])[:3]
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_indices]
selected_voltages = [round(float(voltage_data[i]), 6) for i in lowest_voltage_indices]

# Prepare and print the result in the required format
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print("RESULT_JSON=", RESULT_JSON)