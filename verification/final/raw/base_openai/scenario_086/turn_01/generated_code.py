# required_dependencies: pandas
import andes
import pandas as pd

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True)

# Run power flow
case.run_power_flow()

# Get the phase angle metrics
line_angles = abs(case.Line.angle.v)
line_ids = case.Line.idx.v

# Create a DataFrame to sort the lines
lines_df = pd.DataFrame({
    'line_id': line_ids,
    'absolute_angle': line_angles
})

# Get the top 3 lines by absolute sending-end phase angles
top_lines = lines_df.nlargest(3, 'absolute_angle')

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "selected_line_ids": top_lines['line_id'].astype(int).tolist(),
    "selected_line_metrics": [round(angle, 6) for angle in top_lines['absolute_angle'].tolist()]
}

# Print the RESULT_JSON
print(f"RESULT_JSON={RESULT_JSON}")