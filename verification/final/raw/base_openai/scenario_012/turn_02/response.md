```python
# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
load_idx = case.PQ.add(bus=4, p0=0.014, q0=0.008, name='PQ_VERIFY_012_D')

# Rerun the power flow
case.run()

# Get the slack bus and its voltage
slack_bus = case.Bus.idx.v[case.Slack.idx.v[0]]
slack_voltage = round(float(case.Bus.v.v[case.Slack.idx.v[0]]), 6)

# Get all bus voltages and find those below 1.000 p.u.
voltages = case.Bus.v.v
threshold = 1.0
selected_bus_ids = [int(case.Bus.idx.v[i]) for i in range(len(voltages)) if voltages[i] < threshold]
selected_count = len(selected_bus_ids)

# Create the result JSON object
RESULT_JSON = {
    "added_load_idx": int(load_idx),
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```