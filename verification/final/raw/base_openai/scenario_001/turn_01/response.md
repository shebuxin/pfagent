```python
# required_dependencies: andes
import andes
import json
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage and its index
slack_bus_idx = ssa.slack.bus.v[0]  # Assuming there is at least one slack bus
slack_voltage = ssa.bus.v[slack_bus_idx]

# Get all bus voltages and their IDs
voltages = ssa.bus.v
bus_ids = np.arange(len(voltages))

# Get top-2 highest voltages and their respective bus IDs
sorted_indices = np.argsort(voltages)[::-1][:2]
selected_bus_ids = bus_ids[sorted_indices].tolist()
selected_voltages = np.round(voltages[sorted_indices].tolist(), 6)

# Prepare the result JSON
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")
```