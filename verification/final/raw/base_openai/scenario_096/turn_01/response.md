```python
# required_dependencies: numpy,andes
import numpy as np
import andes

# Load the case
case = "verify_pjm5_096.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get the absolute sending-end phase angles and line IDs
line_phase_angles = np.abs(ssa.Line.phi_s.v)
line_ids = ssa.Line.idx.v

# Get the top-5 lines by absolute sending-end phase angle
top_lines_idx = np.argsort(line_phase_angles)[-5:]
selected_line_ids = line_ids[top_lines_idx].astype(int).tolist()
selected_line_metrics = np.round(line_phase_angles[top_lines_idx], 6).tolist()

# Prepare the result JSON
RESULT_JSON = {
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```