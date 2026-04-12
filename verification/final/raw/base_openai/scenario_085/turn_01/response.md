```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Extract line data
line_ids = ssa.Line.idx.v
sending_end_angle = ssa.Line.angle1.v

# Compute absolute sending-end phase angles and sort
abs_sending_angles = [abs(angle) for angle in sending_end_angle]
sorted_indices = sorted(range(len(abs_sending_angles)), key=lambda i: abs_sending_angles[i], reverse=True)[:2]

# Prepare the result
selected_line_ids = [int(line_ids[i]) for i in sorted_indices]
selected_line_metrics = [round(abs_sending_angles[i], 6) for i in sorted_indices]

# Create the result JSON object
RESULT_JSON = {
    'selected_line_ids': selected_line_ids,
    'selected_line_metrics': selected_line_metrics
}

print(f"RESULT_JSON={RESULT_JSON}")
```