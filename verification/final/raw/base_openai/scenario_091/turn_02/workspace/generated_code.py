
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

# Load PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v
ssb_voltage_target = 1.020
ssa.Slack.set_voltage(slack_bus_idx, ssb_voltage_target)

# Run power flow
andes.run_power_flow(ssa)

# Extracting slack bus voltage and line metrics
slack_voltage = round(ssa.Bus.v[slack_bus_idx][0], 6)
slack_setpoint = ssb_voltage_target

# Get line metrics (sending-end phase angles)
line_angles = ssa.Line.angle1.v  # Sending-end angle
line_ids = ssa.Line.idx.v

# Pair line IDs with their angles and sort
line_metrics = sorted(zip(line_ids, line_angles), key=lambda x: abs(x[1]), reverse=True)[:4]

selected_line_ids = [line_metric[0] for line_metric in line_metrics]
selected_line_metrics = [round(line_metric[1], 6) for line_metric in line_metrics]

# Prepare JSON output
result_json = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")