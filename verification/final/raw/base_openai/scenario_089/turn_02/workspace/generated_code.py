
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

# required_dependencies: andes, json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_idx = case.Slack.idx.v[0]
case.Slack.set("V_target", 1.000)

# Rerun power flow
case.solve()

# Get line metrics for sending-end phase angles
lines = case.Line
line_angles = lines.angle1.v  # Assuming this is for sending-end angle

# Get top-2 lines by absolute sending-end phase angle
top_indices = abs(line_angles).argsort()[-2:][::-1]
selected_line_ids = lines.idx.v[top_indices].tolist()
selected_line_metrics = [{"id": int(selected_line_ids[i]), "angle": round(line_angles[top_indices[i]], 6)} for i in range(2)]

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": 1.000,
    "slack_voltage": round(case.Slack.v.v[slack_idx], 6),
    "selected_line_ids": selected_line_ids,
    "selected_line_metrics": selected_line_metrics
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")