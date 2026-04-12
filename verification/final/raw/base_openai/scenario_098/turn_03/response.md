```python
# required_dependencies: andes
import andes
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_098.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = np.where(ssa.Bus.slack.v)[0][0]
ssa.Bus.V.mag.v[slack_bus_idx] = 1.010

# Scale every PQ load by 1.040
ssa.PQ.p.v *= 1.040
ssa.PQ.q.v *= 1.040

# Rerun power flow
andes.run_power_flow(ssa)

# Identify lines with absolute sending-end phase angle above 0.100 radians
angle_threshold = 0.100
lines_angle = np.abs(ssa.Line.angle.send.v)
selected_line_indices = np.where(lines_angle > angle_threshold)[0]

selected_line_ids = ssa.Line.idx[selected_line_indices].tolist()
selected_count = len(selected_line_ids)

# Create the result JSON
RESULT_JSON = {
    "slack_setpoint": float(1.010),
    "scale_factor": float(1.040),
    "angle_threshold": float(angle_threshold),
    "selected_line_ids": selected_line_ids,
    "selected_count": int(selected_count)
}

# Print the result JSON
print(f'RESULT_JSON={RESULT_JSON}')
```