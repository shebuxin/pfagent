# required_dependencies: andes,json
import andes
import json

# Load PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v
ssb_voltage_target = 1.020
ssa.Slack.set_voltage(slack_bus_idx, ssb_voltage_target)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.set_p(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor)
ssa.PQ.set_q(ssa.PQ.idx.v, ssa.PQ.q.v * scale_factor)

# Run power flow
andes.run_power_flow(ssa)

# Extract line metrics (sending-end phase angles)
angle_threshold = 0.120
line_angles = ssa.Line.angle1.v  # Sending-end angle
line_ids = ssa.Line.idx.v

# Select lines with angles above the threshold
selected_lines = [(line_ids[i], line_angles[i]) for i in range(len(line_angles)) if abs(line_angles[i]) > angle_threshold]
selected_line_ids = [line[0] for line in selected_lines]
selected_count = len(selected_line_ids)

# Prepare JSON output
result_json = {
    "slack_setpoint": ssb_voltage_target,
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")