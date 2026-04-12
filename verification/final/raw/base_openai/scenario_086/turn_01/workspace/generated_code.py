
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _verification_output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(_verification_output_dir, exist_ok=True)
    _verification_existing = [
        name for name in os.listdir(_verification_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _verification_plot_counter = len(_verification_existing)

    def _verification_safe_show(*args, **kwargs):
        global _verification_plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _verification_plot_counter += 1
            plot_path = os.path.join(_verification_output_dir, f"plot_{_verification_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _verification_safe_show
except Exception:
    pass

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