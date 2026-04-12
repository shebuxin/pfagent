
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

# Load the Kundur full case
case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(case)

# Retrieve slack bus voltage
slack_bus_idx = case.slack.idx.v[0]
slack_voltage = round(case.Bus.v[slack_bus_idx], 6)

# Retrieve all bus voltages and sort by value
bus_voltages = case.Bus.v
sorted_idx = bus_voltages.argsort()[::-1][:3]  # Get indices of top 3 highest voltages
selected_bus_ids = sorted_idx.tolist()
selected_voltages = [round(bus_voltages[idx], 6) for idx in selected_bus_ids]

# Build the result JSON object
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")