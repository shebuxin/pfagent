```python
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

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Rerun power flow
andes.run(ssa)

# Retrieve line phase angles
line_phase_angles = ssa.Line.angle_sending.v

# Find lines whose absolute sending-end phase angle is above 0.120 radians
angle_threshold = 0.120
selected_line_ids = np.where(np.abs(line_phase_angles) > angle_threshold)[0].tolist()
selected_count = len(selected_line_ids)

# Construct and print the result JSON
RESULT_JSON = {
    "slack_setpoint": slack_voltage_target,
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}
print(f"RESULT_JSON={RESULT_JSON}")
```