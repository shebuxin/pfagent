
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

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.set('P', ssa.PQ.power.v * scale_factor)
ssa.PQ.set('Q', ssa.PQ.react.v * scale_factor)

# Run the power flow
andes.run_power_flow(ssa)

# Define the angle threshold
angle_threshold = 0.120

# Extract sending-end phase angle for lines
sending_angle = np.abs(ssa.Line.angle.s.v)  # Absolute sending-end angles
line_ids = ssa.Line.idx.v  # Line IDs

# Find lines with absolute sending-end phase angle above the threshold
selected_indices = np.where(sending_angle > angle_threshold)[0]
selected_line_ids = line_ids[selected_indices].tolist()
selected_count = len(selected_line_ids)

# Prepare result in JSON format
RESULT_JSON = {
    "scale_factor": round(scale_factor, 6),
    "angle_threshold": round(angle_threshold, 6),
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Output the result
print(f"RESULT_JSON={RESULT_JSON}")