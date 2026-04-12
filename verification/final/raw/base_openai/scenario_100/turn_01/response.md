```python
# required_dependencies: andes
import andes
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_100.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract voltage data
slack_bus = ssa.Slack.idx.v[0]  # Slack bus index
slack_voltage = round(ssa.Slack.V.v[0], 6)  # Slack bus voltage

# Maximum and minimum voltage bus calculations
max_voltage = round(ssa.Bus.V.v.max(), 6)
min_voltage = round(ssa.Bus.V.v.min(), 6)

max_bus = ssa.Bus.idx.v[ssa.Bus.V.v.argmax()]  # Bus index with max voltage
min_bus = ssa.Bus.idx.v[ssa.Bus.V.v.argmin()]  # Bus index with min voltage

# Prepare results in JSON format
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```