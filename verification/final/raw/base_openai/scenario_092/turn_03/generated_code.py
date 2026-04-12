# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = case.bus.idx.slack
case.slack.set_voltage_target(slack_bus_idx, 1.030)

# Scale every PQ load by 1.060
scaled_load = case.PQ.p * 1.060
case.PQ.set_p(scaled_load)

# Rerun power flow
case.run_power_flow()

# Extract line angles and determine selected lines
line_angles = abs(case.Line.send_angle.v)
angle_threshold = 0.150
selected_line_ids = [i for i, angle in enumerate(line_angles) if angle > angle_threshold]
selected_count = len(selected_line_ids)

# Prepare the result JSON
RESULT_JSON = {
    "slack_setpoint": 1.030,
    "scale_factor": 1.060,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count,
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")