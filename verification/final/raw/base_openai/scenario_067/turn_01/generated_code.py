# required_dependencies: andes
import andes

# Load the Kundur built-in case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa, log=False)

# Retrieve bus voltage values
voltages = ssa.Bus.voltage.v
buses = ssa.Bus.idx.v

# Find max and min voltages and their corresponding bus indices
max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

max_bus = int(buses[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)
min_bus = int(buses[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Output results in the required JSON format
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")