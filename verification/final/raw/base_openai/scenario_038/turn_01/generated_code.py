# required_dependencies: andes
import andes
import numpy as np

# Load IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True)

# Run power flow
case.run()

# Inspect results
bus_voltage = case.V[:]
max_idx = np.argmax(bus_voltage)
min_idx = np.argmin(bus_voltage)

# Prepare result JSON
result_json = {
    "max_bus": int(case.bus[max_idx]),
    "max_voltage": round(float(bus_voltage[max_idx]), 6),
    "min_bus": int(case.bus[min_idx]),
    "min_voltage": round(float(bus_voltage[min_idx]), 6)
}

# Print the result
print("RESULT_JSON=", result_json)