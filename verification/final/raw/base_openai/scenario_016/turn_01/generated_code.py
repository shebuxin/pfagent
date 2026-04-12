# required_dependencies: pandas,json,andes
import pandas as pd
import json
import andes

case = "verify_ieee14_016.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow()

# Get the slack bus and its voltage
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v]
slack_voltage = round(ssa.Bus.V.v[slack_bus_idx], 6)

# Get the top 3 highest voltage buses
voltages = ssa.Bus.V.v
buses = ssa.Bus.idx.v
sorted_indices = voltages.argsort()[::-1][:3]

selected_bus_ids = buses[sorted_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in sorted_indices]

# Prepare the result
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": float(slack_voltage),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")