```python
# required_dependencies: andes
import andes
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_047.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.BUS.idx.v[ssa.BUS.slack.v][0]  # Get the first slack bus ID
slack_setpoint = 1.015
ssa.Y[v] = slack_setpoint  # Setting the slack bus voltage target

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage and count buses below 0.950 p.u.
slack_voltage = round(ssa.BUS.vm.v[ssa.BUS.idx.v == slack_bus][0], 6)
selected_count = (ssa.BUS.vm.v < 0.950).sum()

# Prepare the result JSON object
RESULT_JSON = {
    "slack_bus": slack_bus,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_count": selected_count,
}

# Print the result
print(f'RESULT_JSON={RESULT_JSON}')
```