```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = case.bus.idx.slack
case.slack.set_voltage_target(slack_bus_idx, 1.030)

# Rerun power flow
case.run_power_flow()

# Extract slack voltage and line angles
slack_voltage = round(case.bus.v[slack_bus_idx], 6)
line_angles = abs(case.Line.send_angle.v)

# Get top 5 lines by absolute sending-end phase angle
top_5_lines = sorted(enumerate(line_angles), key=lambda x: x[1], reverse=True)[:5]
selected_line_ids = [line[0] for line in top_5_lines]
selected_line_metrics = [round(line[1], 6) for line in top_5_lines]

# Prepare the result JSON
RESULT_JSON = {
    "slack_setpoint": 1.030,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics,
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")
```