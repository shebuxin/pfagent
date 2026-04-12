```python
# required_dependencies: andes,json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Get line metrics
line_sending_angles = ssa.Line.phi1.v

# Create a list of tuples containing line IDs and their sending-end phase angles
line_metrics = [(idx, abs(angle)) for idx, angle in enumerate(line_sending_angles)]

# Sort by absolute sending-end phase angle in descending order and select top-5
top_lines = sorted(line_metrics, key=lambda x: x[1], reverse=True)[:5]

# Prepare the results
selected_line_ids = [line[0] for line in top_lines]
selected_line_metrics = [round(line[1], 6) for line in top_lines]

# Print the final JSON result
RESULT_JSON = json.dumps({
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
})
print(f"RESULT_JSON={RESULT_JSON}")
```