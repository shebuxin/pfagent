
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

# required_dependencies: numpy, pandas, json
import os
import numpy as np
import pandas as pd
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Extracting bus voltages
bus_voltages = ssa.Bus.v.v  # Assuming this contains the bus voltage values
threshold = 1.050

# Finding buses above threshold
selected_bus_ids = np.where(bus_voltages > threshold)[0].tolist()
selected_count = len(selected_bus_ids)

# Finding the two lowest voltage buses
lowest_bus_ids = np.argsort(bus_voltages)[:2].tolist()
lowest_voltages = [round(bus_voltages[i], 6) for i in lowest_bus_ids]

# Constructing the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages,
}

# Printing the result
print("RESULT_JSON=" + json.dumps(result_json))