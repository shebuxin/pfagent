
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
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_095.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.load.v *= scale_factor

# Rerun power flow
andes.run_power_flow()

# Get absolute sending-end phase angles for lines
line_angles = ssa.Line.angle_send.v
line_ids = ssa.Line.idx.v

# Define the angle threshold
angle_threshold = 0.120

# Collect lines exceeding the angle threshold
selected_lines = [(line_ids[i], abs(line_angles[i])) for i in range(len(line_angles)) if abs(line_angles[i]) > angle_threshold]

# Prepare the output structure
selected_line_ids = [line_id for line_id, _ in selected_lines]
selected_count = len(selected_line_ids)

# Print the result in the required format
RESULT_JSON = {
    "scale_factor": scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}
print("RESULT_JSON=", RESULT_JSON)