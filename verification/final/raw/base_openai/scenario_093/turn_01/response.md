```python
# required_dependencies: andes
import andes
import os
import numpy as np

# Load the uploaded case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_pjm5_093.json")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa, log=False)

# Get the sending-end phase angles for each line
line_angles = np.abs(ssa.Line.angle.v)  # use absolute values
line_ids = ssa.Line.idx.v                # retrieve line ids
 
# Create a list of tuples (line_id, angle)
line_metrics = [(line_id, angle) for line_id, angle in zip(line_ids, line_angles)]
# Sort the lines by angle in descending order
sorted_lines = sorted(line_metrics, key=lambda x: x[1], reverse=True)

# Select the top-2 lines
selected_lines = sorted_lines[:2]

# Prepare the RESULT_JSON
selected_line_ids = [line[0] for line in selected_lines]
selected_line_metrics = [round(line[1], 6) for line in selected_lines]

RESULT_JSON = {
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```