
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
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_048.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_setpoint = 1.020
slack_bus = ssa.Slack.idx.v[0]  # Assuming there's at least one slack bus
ssa.Slack.v_setpoint.set(slack_setpoint)

# Rerun power flow
andes.run_pf(ssa)

# Get slack bus voltage
slack_voltage = ssa.Slack.v.v[0]  # Get the voltage of the slack bus

# Count how many buses fall below 0.960 p.u.
below_threshold = np.where(ssa.Bus.v < 0.960)[0]  # Get indices of buses below threshold
selected_count = len(below_threshold)  # Count of buses below threshold

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "slack_bus": int(slack_bus),  # Convert to int
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")