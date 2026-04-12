
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

# required_dependencies: pandas, numpy
import andes
import numpy as np
import pandas as pd

# Load the IEEE 14 full case
ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Slack.idx.v[0]
slack_setpoint = 1.035
ssa.Slack.v_target.set(slack_setpoint)

# Run power flow analysis
andes.run(ssa)

# Get the slack bus voltage and count buses below 1.010 p.u.
slack_voltage = ssa.Slack.v.v[0]
voltage_levels = ssa.Bus.v
selected_count = np.sum(voltage_levels < 1.010)

# Prepare the JSON result
RESULT_JSON = {
    "slack_bus": slack_bus,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")