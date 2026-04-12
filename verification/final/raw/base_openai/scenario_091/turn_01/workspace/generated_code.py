
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

# Run power flow
andes.run_power_flow(ssa)

# Extract voltage information
voltages = ssa.Bus.v
slack_bus_idx = ssa.Slack.idx.v
slack_bus = ssa.Slack.bus.v[0]
slack_voltage = round(voltages[slack_bus_idx][0], 6)

max_bus_idx = voltages.argmax()
max_bus = ssa.Bus.idx.v[max_bus_idx]
max_voltage = round(voltages[max_bus_idx], 6)

min_bus_idx = voltages.argmin()
min_bus = ssa.Bus.idx.v[min_bus_idx]
min_voltage = round(voltages[min_bus_idx], 6)

# Prepare JSON output
result_json = {
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")