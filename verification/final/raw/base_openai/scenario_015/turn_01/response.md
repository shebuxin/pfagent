```python
# required_dependencies: numpy,pandas
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_015.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.powerflow(ssa)

# Get slack bus information
slack_bus_idx = ssa.bus.v[ssa.bus.slack.v][0]
slack_voltage = ssa.bus.v[slack_bus_idx]

# Get bus voltages and their indices
voltages = ssa.bus.v
bus_ids = np.arange(len(voltages))

# Find the top 2 highest voltages
sorted_indices = np.argsort(voltages)[-2:][::-1]
selected_bus_ids = bus_ids[sorted_indices].tolist()
selected_voltages = voltages[sorted_indices].round(6).tolist()

# Prepare JSON output
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={result_json}")
```