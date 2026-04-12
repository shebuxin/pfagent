# required_dependencies: andes
import andes
import numpy as np

# Load the PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow
andes.run_power_flow(ssa)

# Extract sending-end phase angle for lines
sending_angle = np.abs(ssa.Line.angle.s.v)  # Absolute sending-end angles
line_ids = ssa.Line.idx.v  # Line IDs

# Get top 4 lines by absolute sending-end phase angle
top_indices = np.argsort(sending_angle)[-4:][::-1]  # Indices of top 4, descending
selected_line_ids = line_ids[top_indices].tolist()
selected_line_metrics = [round(sending_angle[idx], 6) for idx in top_indices]

# Prepare result in JSON format
RESULT_JSON = {
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Output the result
print(f"RESULT_JSON={RESULT_JSON}")