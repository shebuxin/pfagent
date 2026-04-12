# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Run power flow
andes.run(ssa, no_output=True)

# Define the threshold
threshold = 1.030

# Count buses above threshold and find their IDs
bus_ids = ssa.Bus.idx.v
bus_voltages = ssa.Bus.v.v
selected_bus_ids = [int(bus_id) for bus_id, voltage in zip(bus_ids, bus_voltages) if voltage > threshold]
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_bus_ids = sorted(zip(bus_ids, bus_voltages), key=lambda x: x[1])[:2]
lowest_bus_ids = [int(bus_id) for bus_id, voltage in lowest_bus_ids]
lowest_voltages = [round(voltage, 6) for bus_id, voltage in lowest_bus_ids]

# Prepare the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")