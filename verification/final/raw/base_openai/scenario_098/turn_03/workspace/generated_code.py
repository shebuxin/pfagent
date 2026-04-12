
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
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_098.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = np.where(ssa.Bus.slack.v)[0][0]
ssa.Bus.V.mag.v[slack_bus_idx] = 1.010

# Scale every PQ load by 1.040
ssa.PQ.p.v *= 1.040
ssa.PQ.q.v *= 1.040

# Rerun power flow
andes.run_power_flow(ssa)

# Identify lines with absolute sending-end phase angle above 0.100 radians
angle_threshold = 0.100
lines_angle = np.abs(ssa.Line.angle.send.v)
selected_line_indices = np.where(lines_angle > angle_threshold)[0]

selected_line_ids = ssa.Line.idx[selected_line_indices].tolist()
selected_count = len(selected_line_ids)

# Create the result JSON
RESULT_JSON = {
    "slack_setpoint": float(1.010),
    "scale_factor": float(1.040),
    "angle_threshold": float(angle_threshold),
    "selected_line_ids": selected_line_ids,
    "selected_count": int(selected_count)
}

# Print the result JSON
print(f'RESULT_JSON={RESULT_JSON}')