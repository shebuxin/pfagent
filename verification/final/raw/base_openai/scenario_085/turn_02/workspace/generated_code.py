
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