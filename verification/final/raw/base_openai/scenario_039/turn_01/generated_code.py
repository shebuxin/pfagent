# required_dependencies: andes
import andes

# Load the IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Run power flow analysis
case.solve_power_flow()

# Extract voltage information
voltages = case.Bus.v
buses = case.Bus.idx

max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

max_bus = int(buses[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)
min_bus = int(buses[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Create the JSON result
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")