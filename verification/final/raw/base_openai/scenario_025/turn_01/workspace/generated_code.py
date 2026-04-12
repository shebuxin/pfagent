
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

# required_dependencies: pandas, numpy, andes
import pandas as pd
import numpy as np
import andes

# Load the case
case = "verify_ieee14_025.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_powerflow(ssa)

# Get all bus voltages
voltages = ssa.Bus.v.v

# Find maximum and minimum voltage buses
max_voltage = np.max(voltages)
min_voltage = np.min(voltages)

max_bus = np.where(voltages == max_voltage)[0][0] + 1  # Convert index to bus number (1-based)
min_bus = np.where(voltages == min_voltage)[0][0] + 1  # Convert index to bus number (1-based)

# Prepare results
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6)
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")