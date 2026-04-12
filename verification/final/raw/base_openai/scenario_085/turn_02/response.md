```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor

# Run power flow
andes.run_power_flow(ssa)

# Extract line data
line_ids = ssa.Line.idx.v
sending_end_angle = ssa.Line.angle1.v

# Identify lines with absolute sending-end phase angle above 0.080 radians
angle_threshold = 0.080
selected_line_ids = [int(line_ids[i]) for i in range(len(sending_end_angle)) if abs(sending_end_angle[i]) > angle_threshold]
selected_count = len(selected_line_ids)

# Create the result JSON object
RESULT_JSON = {
    'scale_factor': scale_factor,
    'angle_threshold': angle_threshold,
    'selected_line_ids': selected_line_ids,
    'selected_count': selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")
```