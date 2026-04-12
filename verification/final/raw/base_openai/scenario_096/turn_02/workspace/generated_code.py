
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

# required_dependencies: numpy,andes
import numpy as np
import andes

# Load the case
case = "verify_pjm5_096.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale all PQ loads by 1.060
scale_factor = 1.060
ssa.PQ.set(ssa.PQ.p.v * scale_factor, ssa.PQ.q.v * scale_factor)

# Run power flow
andes.run_power_flow(ssa)

# Define the angle threshold
angle_threshold = 0.150

# Get the absolute sending-end phase angles and line IDs
line_phase_angles = np.abs(ssa.Line.phi_s.v)
line_ids = ssa.Line.idx.v

# Select lines whose absolute sending-end phase angle is above the threshold
selected_lines_mask = line_phase_angles > angle_threshold
selected_line_ids = line_ids[selected_lines_mask].astype(int).tolist()
selected_count = len(selected_line_ids)

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")