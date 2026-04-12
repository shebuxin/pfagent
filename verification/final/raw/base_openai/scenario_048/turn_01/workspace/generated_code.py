
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
import os
import andes
import numpy as np
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_048.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_pf(ssa)

# Define the voltage threshold
threshold = 1.030

# Count all buses above the threshold
above_threshold = np.where(ssa.Bus.v > threshold)[0]  # Get indices of buses above threshold
selected_bus_ids = above_threshold.tolist()  # Convert to list of bus indices
selected_count = len(selected_bus_ids)  # Count of buses above threshold

# Find the two lowest-voltage buses
voltages = ssa.Bus.v.tolist()
lowest_voltages_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])[:2]  # Get indices of two lowest voltages
lowest_bus_ids = [i for i in lowest_voltages_indices]

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": [round(voltages[i], 6) for i in lowest_bus_ids]
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")