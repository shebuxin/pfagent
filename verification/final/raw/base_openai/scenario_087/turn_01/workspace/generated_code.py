
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