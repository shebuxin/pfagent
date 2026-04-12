# required_dependencies: numpy, pandas, andes
import numpy as np
import pandas as pd
import andes

# Load the built-in IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load before setup at bus 4
bus_idx = np.where(case.Bus.idx == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 not found.")
bus_idx = bus_idx[0]

case.PQ.add('PQ_VERIFY_040_D', bus=bus_idx, p0=0.014, q0=0.008)

# Run power flow
andes.run_power_flow(case)

# Report the slack bus voltage and buses below the threshold
slack_bus = case.Slack.idx[0]
slack_voltage = case.Bus.v[case.Slack.idx[0]]
threshold = 0.950

# Find buses below the threshold
below_threshold_indices = np.where(case.Bus.v < threshold)[0]
selected_bus_ids = case.Bus.idx[below_threshold_indices].tolist()
selected_count = len(selected_bus_ids)

# Prepare the result JSON
RESULT_JSON = {
    'added_load_idx': 'PQ_VERIFY_040_D',
    'slack_bus': slack_bus,
    'slack_voltage': round(slack_voltage, 6),
    'threshold': threshold,
    'selected_bus_ids': selected_bus_ids,
    'selected_count': selected_count
}

# Print the result JSON
print("RESULT_JSON=", RESULT_JSON)