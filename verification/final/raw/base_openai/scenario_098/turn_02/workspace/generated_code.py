
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

# Rerun power flow
andes.run_power_flow(ssa)

# Get slack bus voltage after the run
slack_voltage = round(ssa.Bus.V.mag.v[slack_bus_idx], 6)

# Get top-3 lines by absolute sending-end phase angle
lines_angle = np.abs(ssa.Line.angle.send.v)
top_line_indices = np.argsort(lines_angle)[-3:]

selected_line_ids = ssa.Line.idx[top_line_indices].tolist()
selected_line_metrics = [{
    "line_id": int(ssa.Line.idx[v]),
    "sending_angle": round(ssa.Line.angle.send.v[v], 6),
    "receiving_angle": round(ssa.Line.angle.recv.v[v], 6)
} for v in top_line_indices]

# Create the result JSON
RESULT_JSON = {
    "slack_setpoint": float(1.010),
    "slack_voltage": float(slack_voltage),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result JSON
print(f'RESULT_JSON={RESULT_JSON}')