
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
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.010
slack_idx = ssa.Bus.idx.slack
ssa.Slack.bus_v.target.v[slack_idx] = 1.010

# Rerun power flow
andes.run_powerflow(ssa)

# Extract slack bus voltage and sending-end phase angles of lines
slack_voltage = round(ssa.Bus.V.v[slack_idx], 6)
line_angles = ssa.Line.phi1.v

# Get the absolute values of the line angles
line_angle_abs = abs(line_angles)

# Get the top-3 lines by absolute sending-end phase angle
top_3_indices = line_angle_abs.argsort()[-3:][::-1]
selected_line_ids = ssa.Line.idx.v[top_3_indices].tolist()
selected_line_metrics = line_angles[top_3_indices].round(6).tolist()

# Prepare the result
result = {
    "slack_setpoint": 1.010,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result in the required format
print(f"RESULT_JSON={result}")