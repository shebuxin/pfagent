```python
# required_dependencies: andes, json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Run power flow
case.solve()

# Get voltage information
voltages = case.PQ.v
buses = case.PQ.bus.v

# Find max and min voltage
max_idx = voltages.argmax()
min_idx = voltages.argmin()
slack_idx = case.Slack.idx.v[0]  # Assuming there is only one slack bus

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_bus": int(slack_idx),
    "slack_voltage": round(case.PQ.v[slack_idx], 6),
    "max_bus": int(buses[max_idx]),
    "max_voltage": round(voltages[max_idx], 6),
    "min_bus": int(buses[min_idx]),
    "min_voltage": round(voltages[min_idx], 6)
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")
```