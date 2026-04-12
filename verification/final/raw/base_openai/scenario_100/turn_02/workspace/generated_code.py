
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
case = os.path.join(script_dir, "verify_pjm5_100.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v[0]
ssa.Slack.V.set(1.030)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract the updated slack bus voltage
slack_voltage = round(ssa.Slack.V.v[0], 6)
slack_setpoint = 1.030

# Get lines and their sending-end phase angles
line_phase_angles = abs(ssa.Line.angle_send.v)
lines_with_angles = list(zip(ssa.Line.idx.v, line_phase_angles))

# Sort lines by absolute sending-end phase angle and select top 5
top_lines = sorted(lines_with_angles, key=lambda x: x[1], reverse=True)[:5]

# Prepare selected line IDs and metrics
selected_line_ids = [int(line[0]) for line in top_lines]
selected_line_metrics = [round(line[1], 6) for line in top_lines]

# Prepare results in JSON format
RESULT_JSON = {
    "slack_setpoint": float(slack_setpoint),
    "slack_voltage": float(slack_voltage),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)