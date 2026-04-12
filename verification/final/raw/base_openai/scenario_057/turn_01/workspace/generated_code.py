
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
case = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow
andes.run_power_flow(ssa)

# Gather voltage data
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find the highest voltage buses (excluding slack bus)
voltage_bus_pairs = sorted(zip(bus_ids, bus_voltages), key=lambda x: x[1], reverse=True)

# Identify the slack bus voltage
slack_bus_id = ssa.slack.idx
slack_voltage = bus_voltages[slack_bus_id]

# Get the top-2 highest voltage buses
top_buses = voltage_bus_pairs[:2]
selected_bus_ids = [bus[0] for bus in top_buses]
selected_voltages = [round(bus[1], 6) for bus in top_buses]

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": slack_bus_id,
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")