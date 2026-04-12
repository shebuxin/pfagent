# required_dependencies: pandas
import andes
import os

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow analysis
andes.powerflow(ssa)

# Extract bus voltages
voltages = ssa.Bus.v.v

# Find max and min voltage buses
max_voltage = max(voltages)
min_voltage = min(voltages)
max_bus = ssa.Bus.idx.v[voltages.argmax()]
min_bus = ssa.Bus.idx.v[voltages.argmin()]

# Prepare the JSON output
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6)
}

# Print the result
print("RESULT_JSON=" + str(RESULT_JSON))