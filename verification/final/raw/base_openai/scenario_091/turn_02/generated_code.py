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

# Run power flow
andes.run_power_flow(ssa)

# Extracting slack bus voltage and line metrics
slack_voltage = round(ssa.Bus.v[slack_bus_idx][0], 6)
slack_setpoint = ssb_voltage_target

# Get line metrics (sending-end phase angles)
line_angles = ssa.Line.angle1.v  # Sending-end angle
line_ids = ssa.Line.idx.v

# Pair line IDs with their angles and sort
line_metrics = sorted(zip(line_ids, line_angles), key=lambda x: abs(x[1]), reverse=True)[:4]

selected_line_ids = [line_metric[0] for line_metric in line_metrics]
selected_line_metrics = [round(line_metric[1], 6) for line_metric in line_metrics]

# Prepare JSON output
result_json = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")