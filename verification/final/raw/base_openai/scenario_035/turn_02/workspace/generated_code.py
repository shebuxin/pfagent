
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

# Load the case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.030
slack_bus_index = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
ssa.Vref.v[slack_bus_index] = 1.030

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage and count of buses below 0.970 p.u.
slack_voltage = round(ssa.Bus.v.v[slack_bus_index], 6)
selected_count = np.sum(ssa.Bus.v.v < 0.970)

# Prepare the JSON output
RESULT_JSON = {
    "slack_bus": slack_bus_index,
    "slack_setpoint": round(1.030, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

print("RESULT_JSON=", RESULT_JSON)