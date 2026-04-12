
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

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.fpn(ssa)

# Get the slack bus voltage
slack_bus = ssa.Slack.bus.v[0]
slack_voltage = round(ssa.Slack.V.v[0], 6)

# Get all bus voltages and their IDs
bus_voltages = ssa.Bus.V.v
bus_ids = ssa.Bus.idx.v

# Sort buses by voltage to get the top 5 highest
top_indices = bus_voltages.argsort()[-5:][::-1]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = [round(bus_voltages[i], 6) for i in top_indices]

# Prepare the RESULT_JSON
RESULT_JSON = {
    'slack_bus': int(slack_bus),
    'slack_voltage': slack_voltage,
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages
}

# Print the result in the specified format
print(f"RESULT_JSON={RESULT_JSON}")