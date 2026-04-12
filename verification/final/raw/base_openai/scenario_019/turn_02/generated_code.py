# required_dependencies: numpy,pandas,andes
import os
import numpy as np
import pandas as pd
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_019.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the slack bus index
ssa.Bus.v.set(slack_bus_idx, 1.020)

# Rerun power flow
andes.run_power_flow(ssa)

# Report slack bus voltage and count of buses below 1.000 p.u.
slack_voltage = round(ssa.Bus.v.v[slack_bus_idx], 6)
selected_count = np.sum(ssa.Bus.v.v < 1.000)

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_setpoint": 1.020,
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

print("RESULT_JSON=", RESULT_JSON)