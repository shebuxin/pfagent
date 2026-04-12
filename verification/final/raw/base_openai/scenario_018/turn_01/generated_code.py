# required_dependencies: numpy,pandas
import os
import andes
import numpy as np
import pandas as pd

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_018.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Gather voltage data
slack_bus_idx = np.where(ssa.Bus.slack.v)[0][0]
slack_bus = ssa.Bus.idx.v[slack_bus_idx]
slack_voltage = round(ssa.Bus.Vmag.v[slack_bus_idx], 6)

# Get all bus voltages and IDs, sort by voltage
bus_voltages = ssa.Bus.Vmag.v
bus_ids = ssa.Bus.idx.v
sorted_indices = np.argsort(bus_voltages)[::-1][:5]
selected_bus_ids = bus_ids[sorted_indices].tolist()
selected_voltages = np.round(bus_voltages[sorted_indices], 6).tolist()

# Prepare result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)