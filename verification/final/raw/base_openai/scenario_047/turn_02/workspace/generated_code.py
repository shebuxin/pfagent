
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
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_047.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.BUS.idx.v[ssa.BUS.slack.v][0]  # Get the first slack bus ID
slack_setpoint = 1.015
ssa.Y[v] = slack_setpoint  # Setting the slack bus voltage target

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage and count buses below 0.950 p.u.
slack_voltage = round(ssa.BUS.vm.v[ssa.BUS.idx.v == slack_bus][0], 6)
selected_count = (ssa.BUS.vm.v < 0.950).sum()

# Prepare the result JSON object
RESULT_JSON = {
    "slack_bus": slack_bus,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_count": selected_count,
}

# Print the result
print(f'RESULT_JSON={RESULT_JSON}')