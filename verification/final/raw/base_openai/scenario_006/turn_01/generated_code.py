# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True)

# Run power flow
case.run()

# Define the voltage threshold
threshold = 1.020

# Count all buses above threshold
selected_bus_ids = case.Bus.idx.v[case.Bus.v >= threshold].tolist()
selected_count = len(selected_bus_ids)

# Get the lowest two voltage buses
voltages = case.Bus.v.tolist()
lowest_bus_ids = sorted(range(len(voltages)), key=lambda i: voltages[i])[:2]

lowest_voltages = [round(voltages[i], 6) for i in lowest_bus_ids]

# Prepare the result in required JSON format
RESULT_JSON = json.dumps({
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
})

print(f"RESULT_JSON={RESULT_JSON}")