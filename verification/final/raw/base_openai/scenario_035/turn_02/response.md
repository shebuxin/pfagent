```python
# required_dependencies: andes
import andes

# Load the case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.030
slack_bus_index = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
ssa.Vref.v[slack_bus_index] = 1.030

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage and count of buses below 0.970 p.u.
slack_voltage = round(ssa.Bus.v.v[slack_bus_index], 6)
selected_count = np.sum(ssa.Bus.v.v < 0.970)

# Prepare the JSON output
RESULT_JSON = {
    "slack_bus": slack_bus_index,
    "slack_setpoint": round(1.030, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

print("RESULT_JSON=", RESULT_JSON)
```