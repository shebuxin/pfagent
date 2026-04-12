# required_dependencies: andes
import andes
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_100.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v[0]
ssa.Slack.V.set(1.030)

# Scale each PQ load by 1.060
scale_factor = 1.060
ssa.PQ.P.set(ssa.PQ.P.v * scale_factor)
ssa.PQ.Q.set(ssa.PQ.Q.v * scale_factor)

# Run power flow analysis
andes.run_power_flow(ssa)

# Define the angle threshold
angle_threshold = 0.150

# Get lines whose absolute sending-end phase angle is above the threshold
lines_above_threshold = [
    (line.idx.v, line.angle_send.v)
    for line in ssa.Line if abs(line.angle_send.v) > angle_threshold
]

# Prepare selected line IDs
selected_line_ids = [int(line[0]) for line in lines_above_threshold]
selected_count = len(selected_line_ids)

# Prepare results in JSON format
RESULT_JSON = {
    "slack_setpoint": 1.030,
    "scale_factor": float(scale_factor),
    "angle_threshold": float(angle_threshold),
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)