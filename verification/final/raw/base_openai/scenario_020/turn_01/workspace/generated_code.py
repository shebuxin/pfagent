
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

# required_dependencies: pandas
import os
import andes
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_ieee14_020.xlsx")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Define threshold and gather bus voltages
threshold = 1.020
bus_voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Count the number of buses above the threshold
selected_bus_ids = bus_ids[bus_voltages > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
sorted_indices = np.argsort(bus_voltages)
lowest_bus_ids = bus_ids[sorted_indices][:2].tolist()
lowest_voltages = [round(bus_voltages[idx], 6) for idx in sorted_indices[:2]]

# Prepare the result as required
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result in the specified format
print("RESULT_JSON=" + json.dumps(result_json))