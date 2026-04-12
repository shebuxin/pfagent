# required_dependencies: andes
import andes
import numpy as np

# Load the case
case = "verify_pjm5_099.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_idx = np.where(ssa.Bus.type == 1)[0]
slack_voltage_target = 1.020
ssa.Bus.v[slack_idx] = slack_voltage_target

# Rerun power flow
andes.run(ssa)

# Retrieve updated slack bus voltage
slack_voltage = round(ssa.Bus.v[slack_idx][0], 6)

# Get line phase angles and their corresponding indices
line_phase_angles = ssa.Line.angle_sending.v
line_indices = np.arange(len(line_phase_angles))

# Get top-4 lines by absolute sending-end phase angle
top_lines_idx = np.argsort(np.abs(line_phase_angles))[-4:]
selected_line_ids = line_indices[top_lines_idx].tolist()
selected_line_metrics = [{'line_idx': int(idx), 'angle': round(line_phase_angles[idx], 6)} for idx in selected_line_ids]

# Construct and print the result JSON
RESULT_JSON = {
    "slack_setpoint": slack_voltage_target,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}
print(f"RESULT_JSON={RESULT_JSON}")