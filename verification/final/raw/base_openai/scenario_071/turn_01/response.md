```python
# required_dependencies: andes,json
import andes
import json
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_071.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Retrieve bus voltages
voltage_data = ssa.Bus.voltage.v
bus_ids = ssa.Bus.id.v

# Determine the slack bus ID and voltage
slack_bus_idx = np.where(ssa.BUS_TYPE.v == 3)[0][0]  # Common practice for slack busfinding.
slack_bus = bus_ids[slack_bus_idx]
slack_voltage = round(voltage_data[slack_bus_idx], 6)

# Find top 2 highest voltage buses
highest_voltages_idx = np.argsort(voltage_data)[-2:][::-1]  # Get indices of top 2 voltages in descending order
selected_bus_ids = bus_ids[highest_voltages_idx].tolist()
selected_voltages = [round(voltage_data[idx], 6) for idx in highest_voltages_idx]

# Prepare the result
result_json = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "selected_bus_ids": list(map(int, selected_bus_ids)),
    "selected_voltages": selected_voltages
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")
```