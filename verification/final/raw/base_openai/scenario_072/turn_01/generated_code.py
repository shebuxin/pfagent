# required_dependencies: numpy,andes
import numpy as np
import andes
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_072.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus and voltage
slack_bus_idx = ssa.Slack.idx.v[0]
slack_voltage = ssa.Slack.vmag.v[0]

# Get bus voltages
bus_voltages = ssa.Bus.vmag.v
bus_ids = ssa.Bus.idx.v

# Get top 3 highest voltage buses
top_indices = np.argsort(bus_voltages)[-3:][::-1]
selected_bus_ids = bus_ids[top_indices].astype(int).tolist()
selected_voltages = np.round(bus_voltages[top_indices].tolist(), 6)

# Prepare result
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print("RESULT_JSON=", RESULT_JSON)