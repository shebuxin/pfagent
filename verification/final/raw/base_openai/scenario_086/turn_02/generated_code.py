# required_dependencies: pandas
import andes
import pandas as pd

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True)

# Scale every PQ load by 1.040
scale_factor = 1.040
case.PQ.load.v *= scale_factor

# Rerun power flow
case.run_power_flow()

# Get the phase angle metrics
line_angles = abs(case.Line.angle.v)
line_ids = case.Line.idx.v

# Define the angle threshold
angle_threshold = 0.100

# Select lines with absolute sending-end phase angle above the threshold
selected_lines = [(line_id, angle) for line_id, angle in zip(line_ids, line_angles) if angle > angle_threshold]

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": [int(line[0]) for line in selected_lines],
    "selected_count": len(selected_lines)
}

# Print the RESULT_JSON
print(f"RESULT_JSON={RESULT_JSON}")