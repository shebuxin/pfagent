# required_dependencies: andes,json
import andes
import json
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_094.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Get the line data
line_ids = ssa.Line.idx.v
sending_angles = ssa.Line.phi1.v  # Assuming phi1 is the sending end phase angle

# Create a list of tuples (line_id, absolute_angle)
line_metrics = [(line_ids[i], abs(sending_angles[i])) for i in range(len(line_ids))]
# Sort by absolute sending-end phase angle in descending order
line_metrics.sort(key=lambda x: x[1], reverse=True)

# Get the top 3 lines
selected_lines = line_metrics[:3]
selected_line_ids = [line[0] for line in selected_lines]
selected_line_metrics = [round(line[1], 6) for line in selected_lines]

# Prepare the result JSON
RESULT_JSON = {
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")