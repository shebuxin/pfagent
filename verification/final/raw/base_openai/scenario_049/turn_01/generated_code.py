# required_dependencies: andes
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_049.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count buses above 1.040 p.u. and find the two lowest-voltage buses
threshold = 1.040
selected_bus_ids = []
lowest_buses = []
lowest_voltages = []

for idx, voltage in enumerate(ssa.V.v):
    if voltage > threshold:
        selected_bus_ids.append(ssa.Bus.idx.v[idx])
    if len(lowest_buses) < 2:
        lowest_buses.append((ssa.Bus.idx.v[idx], voltage))
    else:
        if voltage < lowest_buses[1][1]:
            lowest_buses[1] = (ssa.Bus.idx.v[idx], voltage)
            lowest_buses.sort(key=lambda x: x[1])  # Sort by voltage

# Prepare the results
lowest_buses_sorted = sorted(lowest_buses, key=lambda x: x[1])[:2]
lowest_bus_ids = [bus[0] for bus in lowest_buses_sorted]
lowest_voltages = [round(bus[1], 6) for bus in lowest_buses_sorted]

# Create the RESULT_JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")