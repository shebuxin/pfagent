
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
import andes
import numpy as np
import pandas as pd

# Load the IEEE 14 full case
ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run(ssa)

# Count all buses above 1.025 p.u.
threshold = 1.025
voltage_levels = ssa.Bus.v
selected_bus_ids = np.where(voltage_levels > threshold)[0].tolist()
selected_count = len(selected_bus_ids)

# Get the two lowest-voltage buses
lowest_voltages_indices = np.argsort(voltage_levels)[:2]
lowest_bus_ids = lowest_voltages_indices.tolist()
lowest_voltages = [round(voltage_levels[i], 6) for i in lowest_voltages_indices]

# Prepare the JSON result
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")