# required_dependencies: numpy, pandas, json
import os
import numpy as np
import pandas as pd
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[np.where(ssa.Bus.type.v == 'slack')[0][0]]  # Assuming only one slack bus
slack_setpoint = 1.035
ssa.VM.set(slack_bus_idx, slack_setpoint)

# Rerun power flow
andes.run(ssa)

# Get slack bus voltage and count buses below 0.980 p.u.
slack_voltage = round(ssa.Bus.v.v[slack_bus_idx], 6)
selected_count = np.sum(ssa.Bus.v.v < 0.980)

# Constructing the result JSON
result_json = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count,
}

# Printing the result
print("RESULT_JSON=" + json.dumps(result_json))