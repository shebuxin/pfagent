```python
# required_dependencies: pandas
import os
import andes
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_074.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Extract voltage data
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.v.v

# Identify slack bus voltage
slack_bus_idx = ssa.Slack.bus.v[0]  # Assuming the first Slack bus
slack_voltage = round(voltages[slack_bus_idx], 6)

# Top 5 highest-voltage buses
top_indices = (-voltages).argsort()[:5]  # Get indices of top 5 voltages
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = [round(voltages[i], 6) for i in top_indices]

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```