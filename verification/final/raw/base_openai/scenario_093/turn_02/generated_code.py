# required_dependencies: andes
import andes
import os
import numpy as np

# Load the uploaded case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_pjm5_093.json")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Run power flow
andes.run_power_flow(ssa, log=False)

# Get the sending-end phase angles for each line
line_angles = np.abs(ssa.Line.angle.v)  # use absolute values
angle_threshold = 0.080  # radians

# Select lines with absolute sending-end phase angle above the threshold
selected_lines = [line_id for line_id, angle in zip(ssa.Line.idx.v, line_angles) if angle > angle_threshold]

# Prepare the RESULT_JSON
selected_count = len(selected_lines)

RESULT_JSON = {
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_lines,
    "selected_count": selected_count
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)