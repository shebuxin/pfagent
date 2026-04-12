```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case 
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.010
slack_idx = ssa.Bus.idx.slack
ssa.Slack.bus_v.target.v[slack_idx] = 1.010

# Rerun power flow
andes.run_powerflow(ssa)

# Extract slack bus voltage and sending-end phase angles of lines
slack_voltage = round(ssa.Bus.V.v[slack_idx], 6)
line_angles = ssa.Line.phi1.v

# Get the absolute values of the line angles
line_angle_abs = abs(line_angles)

# Get the top-3 lines by absolute sending-end phase angle
top_3_indices = line_angle_abs.argsort()[-3:][::-1]
selected_line_ids = ssa.Line.idx.v[top_3_indices].tolist()
selected_line_metrics = line_angles[top_3_indices].round(6).tolist()

# Prepare the result
result = {
    "slack_setpoint": 1.010,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result in the required format
print(f"RESULT_JSON={result}")
```