
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

# required_dependencies: numpy,pandas,andes
import numpy as np
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_052.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
result = andes.run_power_flow(ssa)

# Extract bus voltages
voltages = ssa.Bus.v.v

# Find maximum and minimum voltage buses
max_idx = np.argmax(voltages)
min_idx = np.argmin(voltages)

max_bus = ssa.Bus.idx.v[max_idx]
max_voltage = round(float(voltages[max_idx]), 6)
min_bus = ssa.Bus.idx.v[min_idx]
min_voltage = round(float(voltages[min_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")