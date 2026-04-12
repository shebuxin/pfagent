
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

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.060
load_scale_factor = 1.060
ssa.PQ.P.v *= load_scale_factor
ssa.PQ.Q.v *= load_scale_factor

# Run power flow
andes.run(ssa)

# Get line metrics
line_sending_angles = ssa.Line.phi1.v

# Filter lines based on the absolute sending-end phase angle threshold
angle_threshold = 0.150
selected_lines = [(idx, abs(angle)) for idx, angle in enumerate(line_sending_angles) if abs(angle) > angle_threshold]

# Prepare the results
selected_line_ids = [line[0] for line in selected_lines]
selected_count = len(selected_line_ids)

# Print the final JSON result
RESULT_JSON = json.dumps({
    "scale_factor": load_scale_factor,
    "angle_threshold": angle_threshold,
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
})
print(f"RESULT_JSON={RESULT_JSON}")