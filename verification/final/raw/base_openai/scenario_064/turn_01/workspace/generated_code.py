
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

# required_dependencies: numpy,pandas,json
import os
import andes
import numpy as np
import pandas as pd
import json

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run the power flow
andes.run_power_flow(ssa)

# Set the voltage threshold
threshold = 1.020

# Count all buses above the threshold
buses_above_threshold = np.where(ssa.Bus.v.v > threshold)[0]
selected_bus_ids = buses_above_threshold.tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_voltage_indices = np.argsort(ssa.Bus.v.v)[:2]
lowest_bus_ids = lowest_voltage_indices.tolist()
lowest_voltages = [round(ssa.Bus.v.v[bus_id], 6) for bus_id in lowest_bus_ids]

# Create the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')