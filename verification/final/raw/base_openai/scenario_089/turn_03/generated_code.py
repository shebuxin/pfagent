# required_dependencies: andes, json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_idx = case.Slack.idx.v[0]
case.Slack.set("V_target", 1.000)

# Scale every PQ load by 1.030
scale_factor = 1.030
case.PQ.set("P", case.PQ.P.v * scale_factor)
case.PQ.set("Q", case.PQ.Q.v * scale_factor)

# Rerun power flow
case.solve()

# Find lines with absolute sending-end phase angle above 0.080 radians
angle_threshold = 0.080
lines = case.Line
line_angles = lines.angle1.v  # Assuming this is for sending-end angle

selected_indices = abs(line_angles) > angle_threshold
selected_line_ids = lines.idx.v[selected_indices].tolist()
selected_count = len(selected_line_ids)

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": 1.000,
    "scale_factor": round(scale_factor, 6),
    "angle_threshold": round(angle_threshold, 6),
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")