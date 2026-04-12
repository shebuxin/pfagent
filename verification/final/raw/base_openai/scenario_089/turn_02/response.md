```python
# required_dependencies: andes, json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_idx = case.Slack.idx.v[0]
case.Slack.set("V_target", 1.000)

# Rerun power flow
case.solve()

# Get line metrics for sending-end phase angles
lines = case.Line
line_angles = lines.angle1.v  # Assuming this is for sending-end angle

# Get top-2 lines by absolute sending-end phase angle
top_indices = abs(line_angles).argsort()[-2:][::-1]
selected_line_ids = lines.idx.v[top_indices].tolist()
selected_line_metrics = [{"id": int(selected_line_ids[i]), "angle": round(line_angles[top_indices[i]], 6)} for i in range(2)]

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": 1.000,
    "slack_voltage": round(case.Slack.v.v[slack_idx], 6),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")
```