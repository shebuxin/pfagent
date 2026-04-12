# required_dependencies: pandas
import andes
import os
import json

# Load the Kundur full case
script_dir = os.getcwd()
case = os.path.join(script_dir, "kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Analyze bus voltages
voltages = ssa.Bus.v.val
buses = ssa.Bus.idx.v

# Threshold for bus voltage
threshold = 0.990

# Find buses above threshold
selected_bus_ids = list(buses[voltages > threshold])
selected_count = len(selected_bus_ids)

# Find the two lowest voltage buses
lowest_indices = voltages.argsort()[:2]
lowest_bus_ids = buses[lowest_indices].tolist()
lowest_voltages = [round(voltages[i], 6) for i in lowest_indices]

# Create JSON result
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))