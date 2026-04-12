
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

# Scale each PQ load by 1.040
scale_factor = 1.040
ssa.PQ.set(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor, ssa.PQ.q.v * scale_factor)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get the line data
line_ids = ssa.Line.idx.v
sending_angles = ssa.Line.phi1.v  # Assuming phi1 is the sending end phase angle
angle_threshold = 0.100  # radians

# Create a list of line ids where the absolute sending-end phase angle is above the threshold
selected_lines = [line_ids[i] for i in range(len(line_ids)) if abs(sending_angles[i]) > angle_threshold]

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": round(scale_factor, 6),
    "angle_threshold": round(angle_threshold, 6),
    "selected_line_ids": selected_lines,
    "selected_count": len(selected_lines)
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")