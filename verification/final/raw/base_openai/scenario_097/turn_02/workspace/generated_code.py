
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

# Load the case
case = os.path.join(os.getcwd(), "verify_pjm5_097.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.bus.idx.where(ssa.bus.slack.v)
ssa.bus.v.set(slack_bus, 1.000)

# Rerun power flow
andes.run_power_flow()

# Get slack voltage
slack_voltage = ssa.bus.v[slack_bus].round(6)

# Get the top-2 lines by absolute sending-end phase angle
lines_abs_phase_angle = abs(ssa.Line.angle1.v)
top_line_indices = lines_abs_phase_angle.argsort()[-2:][::-1]  # Get top-2 indices
selected_line_ids = ssa.Line.idx.v[top_line_indices].tolist()
selected_line_metrics = {
    "angle1": [float(ssa.Line.angle1.v[i].round(6)) for i in top_line_indices],
    "angle2": [float(ssa.Line.angle2.v[i].round(6)) for i in top_line_indices],
}

# Construct the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": float(1.000),
    "slack_voltage": float(slack_voltage),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")