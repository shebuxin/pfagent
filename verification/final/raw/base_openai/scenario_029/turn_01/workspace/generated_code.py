
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
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow
andes.run_power_flow(ssa)

# Get slack bus information
slack_bus = ssa.SlackBus.idx.v[0]
slack_voltage = round(ssa.SlackBus.v.v[0], 6)

# Get voltage of all buses
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find the top 2 highest voltage buses
sorted_indices = np.argsort(bus_voltages)[-2:][::-1]  # Sort and get last two in descending order
selected_bus_ids = bus_ids[sorted_indices].tolist()
selected_voltages = [round(bus_voltages[idx], 6) for idx in sorted_indices]

# Prepare RESULT_JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Output the result
print("RESULT_JSON=", RESULT_JSON)