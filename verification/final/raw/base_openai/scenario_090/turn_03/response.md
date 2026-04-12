```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case 
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.010
slack_idx = ssa.Bus.idx.slack
ssa.Slack.bus_v.target.v[slack_idx] = 1.010

# Scale every PQ load by 1.040
scale_factor = 1.040
ssa.PQ.P.v *= scale_factor
ssa.PQ.Q.v *= scale_factor

# Rerun power flow
andes.run_powerflow(ssa)

# Extract sending-end phase angles of lines
line_angles = ssa.Line.phi1.v

# Identify lines whose absolute sending-end phase angle is above 0.100 radians
angle_threshold = 0.100
selected_indices = abs(line_angles) > angle_threshold
selected_line_ids = ssa.Line.idx.v[selected_indices].tolist()
selected_count = len(selected_line_ids)

# Prepare the result
result = {
    "slack_setpoint": 1.010,
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result in the required format
print(f"RESULT_JSON={result}")
```