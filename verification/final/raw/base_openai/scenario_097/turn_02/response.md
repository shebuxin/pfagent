```python
# required_dependencies: andes
import andes
import os

# Load the case
case = os.path.join(os.getcwd(), "verify_pjm5_097.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.bus.idx.where(ssa.bus.slack.v)
ssa.bus.v.set(slack_bus, 1.000)

# Rerun power flow
andes.run_power_flow()

# Get slack voltage
slack_voltage = ssa.bus.v[slack_bus].round(6)

# Get the top-2 lines by absolute sending-end phase angle
lines_abs_phase_angle = abs(ssa.Line.angle1.v)
top_line_indices = lines_abs_phase_angle.argsort()[-2:][::-1]  # Get top-2 indices
selected_line_ids = ssa.Line.idx.v[top_line_indices].tolist()
selected_line_metrics = {
    "angle1": [float(ssa.Line.angle1.v[i].round(6)) for i in top_line_indices],
    "angle2": [float(ssa.Line.angle2.v[i].round(6)) for i in top_line_indices],
}

# Construct the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": float(1.000),
    "slack_voltage": float(slack_voltage),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```