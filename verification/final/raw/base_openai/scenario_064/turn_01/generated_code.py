# required_dependencies: numpy,pandas,json
import os
import andes
import numpy as np
import pandas as pd
import json

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run the power flow
andes.run_power_flow(ssa)

# Set the voltage threshold
threshold = 1.020

# Count all buses above the threshold
buses_above_threshold = np.where(ssa.Bus.v.v > threshold)[0]
selected_bus_ids = buses_above_threshold.tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_voltage_indices = np.argsort(ssa.Bus.v.v)[:2]
lowest_bus_ids = lowest_voltage_indices.tolist()
lowest_voltages = [round(ssa.Bus.v.v[bus_id], 6) for bus_id in lowest_bus_ids]

# Create the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')