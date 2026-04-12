# required_dependencies: pandas, numpy
import os
import andes
import numpy as np
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_048.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_pf(ssa)

# Define the voltage threshold
threshold = 1.030

# Count all buses above the threshold
above_threshold = np.where(ssa.Bus.v > threshold)[0]  # Get indices of buses above threshold
selected_bus_ids = above_threshold.tolist()  # Convert to list of bus indices
selected_count = len(selected_bus_ids)  # Count of buses above threshold

# Find the two lowest-voltage buses
voltages = ssa.Bus.v.tolist()
lowest_voltages_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])[:2]  # Get indices of two lowest voltages
lowest_bus_ids = [i for i in lowest_voltages_indices]

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": [round(voltages[i], 6) for i in lowest_bus_ids]
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")