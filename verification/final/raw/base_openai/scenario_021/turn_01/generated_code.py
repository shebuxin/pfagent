# required_dependencies: andes
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_021.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count buses above the threshold
threshold = 1.025
above_threshold_buses = ssa.Bus.ss[ssa.Bus.v > threshold]
selected_bus_ids = above_threshold_buses.idx.tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest voltage buses
lowest_voltage_buses = ssa.Bus.ss[np.argsort(ssa.Bus.v)][:2]
lowest_bus_ids = lowest_voltage_buses.idx.tolist()
lowest_voltages = lowest_voltage_buses.v.round(6).tolist()

# Prepare the result
result_json = {
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={json.dumps(result_json)}")