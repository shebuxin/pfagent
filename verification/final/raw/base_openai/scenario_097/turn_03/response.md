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

# Scale every PQ load by 1.030
ssa.PQ.p.set(ssa.PQ.idx.v, ssa.PQ.p.v * 1.030)
ssa.PQ.q.set(ssa.PQ.idx.v, ssa.PQ.q.v * 1.030)

# Rerun power flow
andes.run_power_flow()

# Get slack voltage
slack_voltage = ssa.bus.v[slack_bus].round(6)

# Define the angle threshold
angle_threshold = 0.080

# Get all lines whose absolute sending-end phase angle is above the threshold
lines_abs_phase_angle = abs(ssa.Line.angle1.v)
selected_line_indices = np.where(lines_abs_phase_angle > angle_threshold)[0]
selected_line_ids = ssa.Line.idx.v[selected_line_indices].tolist()
selected_count = len(selected_line_ids)

# Construct the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": float(1.000),
    "scale_factor": float(1.030),
    "angle_threshold": float(angle_threshold),
    "selected_line_ids": selected_line_ids,
    "selected_count": int(selected_count)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```