
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

# required_dependencies: pandas
import os
import andes
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_022.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.040
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v]
slack_setpoint = 1.040
ssa.Bus.v.set(slack_bus_idx, slack_setpoint)

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage
slack_voltage = round(ssa.Bus.v.v[slack_bus_idx][0], 6)

# Count buses below the threshold of 1.015 p.u.
selected_count = np.sum(ssa.Bus.v.v < 1.015)

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx[0]),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(RESULT_JSON))