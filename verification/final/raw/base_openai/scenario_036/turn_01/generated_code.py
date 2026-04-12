# required_dependencies: andes
import andes
import numpy as np
import json

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Define the voltage threshold
threshold = 1.050

# Count buses above the threshold
voltage_levels = ssa.Bus.v
selected_bus_ids = np.where(voltage_levels > threshold)[0].tolist()
selected_count = len(selected_bus_ids)

# Get the two lowest-voltage buses
sorted_buses = np.argsort(voltage_levels)
lowest_bus_ids = sorted_buses[:2].tolist()
lowest_voltages = [round(voltage_levels[idx], 6) for idx in lowest_bus_ids]

# Prepare the JSON result
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result in the specified format
print(f"RESULT_JSON={json.dumps(result_json)}")