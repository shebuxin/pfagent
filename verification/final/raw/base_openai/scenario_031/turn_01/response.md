```python
# required_dependencies: andes,json
import andes
import json
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Find the slack bus and its voltage
slack_bus_idx = ssa.Slack.idx.v[0]
slack_voltage = round(ssa.Bus.Vmag.v[slack_bus_idx], 6)

# Get all bus voltages and their indices
voltages = ssa.Bus.Vmag.v
bus_ids = ssa.Bus.idx.v

# Select the top 4 highest-voltage buses
top_4_indices = np.argsort(voltages)[-4:][::-1]
selected_bus_ids = bus_ids[top_4_indices].astype(int).tolist()
selected_voltages = np.round(voltages[top_4_indices], 6).tolist()

# Construct the result JSON
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))
```