```python
# required_dependencies: andes,json
import andes
import json

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get the slack bus index and its voltage
slack_bus_idx = ssa.PQ.idx.v[ssa.Slack.bus.v[0]]
slack_voltage = round(ssa.PQ.v.v[slack_bus_idx], 6)

# Get all bus voltages
bus_voltages = ssa.PQ.v.v
bus_ids = ssa.PQ.bus.v

# Find the top-5 highest voltages
sorted_indices = bus_voltages.argsort()[::-1]
top_bus_ids = bus_ids[sorted_indices[:5]].tolist()
top_voltages = [round(bus_voltages[i], 6) for i in sorted_indices[:5]]

# Prepare the result JSON
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": top_bus_ids,
    "selected_voltages": top_voltages
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))
```