
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

# Scale every PQ load by 1.030
scale_factor = 1.030
case.PQ.set("P", case.PQ.P.v * scale_factor)
case.PQ.set("Q", case.PQ.Q.v * scale_factor)

# Rerun power flow
case.solve()

# Find lines with absolute sending-end phase angle above 0.080 radians
angle_threshold = 0.080
lines = case.Line
line_angles = lines.angle1.v  # Assuming this is for sending-end angle

selected_indices = abs(line_angles) > angle_threshold
selected_line_ids = lines.idx.v[selected_indices].tolist()
selected_count = len(selected_line_ids)

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_setpoint": 1.000,
    "scale_factor": round(scale_factor, 6),
    "angle_threshold": round(angle_threshold, 6),
    "selected_line_ids": selected_line_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")