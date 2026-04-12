
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

# required_dependencies: numpy, pandas, andes
import numpy as np
import pandas as pd
import andes

# Load the built-in IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(case)

# Extract bus voltage data and find the two lowest voltage buses
voltage_data = case.Bus.v
bus_ids = case.Bus.idx

# Get indices of the two lowest voltages
lowest_voltage_indices = np.argsort(voltage_data)[:2]

# Prepare the result JSON
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = [round(voltage_data[idx], 6) for idx in lowest_voltage_indices]

# Print the result JSON
RESULT_JSON = {'selected_bus_ids': selected_bus_ids, 'selected_voltages': selected_voltages}
print("RESULT_JSON=", RESULT_JSON)