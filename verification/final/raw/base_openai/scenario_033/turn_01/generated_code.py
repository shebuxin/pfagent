# required_dependencies: numpy,json
import andes
import numpy as np
import json
import os

# Load the built-in IEEE 39 case
case_path = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Count buses above the voltage threshold
threshold = 1.020
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_indices = np.argsort(ssa.Bus.v.v)[:2]
lowest_bus_ids = ssa.Bus.idx.v[lowest_indices].tolist()
lowest_voltages = np.round(ssa.Bus.v.v[lowest_indices], 6).tolist()

# Prepare the result JSON
result_json = {
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages,
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(result_json))