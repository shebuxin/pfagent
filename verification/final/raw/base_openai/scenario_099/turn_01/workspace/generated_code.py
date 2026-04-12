
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
case = "verify_pjm5_099.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Retrieve voltage data
voltages = ssa.Bus.v
slack_idx = ssa.Bus.type == 1  # Slack bus type
max_idx = voltages == max(voltages)
min_idx = voltages == min(voltages)

# Prepare results
slack_bus = ssa.Bus.idx[slack_idx].tolist()[0]
slack_voltage = round(voltages[slack_idx][0], 6)
max_bus = ssa.Bus.idx[max_idx].tolist()[0]
max_voltage = round(voltages[max_idx][0], 6)
min_bus = ssa.Bus.idx[min_idx].tolist()[0]
min_voltage = round(voltages[min_idx][0], 6)

# Construct and print the result JSON
RESULT_JSON = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}
print(f"RESULT_JSON={RESULT_JSON}")