```python
# required_dependencies: andes
import andes

# Load the case
case = "verify_pjm5_099.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Retrieve voltage data
voltages = ssa.Bus.v
slack_idx = ssa.Bus.type == 1  # Slack bus type
max_idx = voltages == max(voltages)
min_idx = voltages == min(voltages)

# Prepare results
slack_bus = ssa.Bus.idx[slack_idx].tolist()[0]
slack_voltage = round(voltages[slack_idx][0], 6)
max_bus = ssa.Bus.idx[max_idx].tolist()[0]
max_voltage = round(voltages[max_idx][0], 6)
min_bus = ssa.Bus.idx[min_idx].tolist()[0]
min_voltage = round(voltages[min_idx][0], 6)

# Construct and print the result JSON
RESULT_JSON = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}
print(f"RESULT_JSON={RESULT_JSON}")
```