# required_dependencies: pandas
import os
import andes
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_022.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Define threshold
threshold = 1.030

# Find buses above the threshold
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_bus_ids = ssa.Bus.idx.v[np.argsort(ssa.Bus.v.v)[:2]].tolist()
lowest_voltages = np.round(ssa.Bus.v.v[lowest_bus_ids], 6).tolist()

# Prepare the result JSON
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(RESULT_JSON))