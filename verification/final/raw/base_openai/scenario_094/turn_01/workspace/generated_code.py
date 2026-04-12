
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