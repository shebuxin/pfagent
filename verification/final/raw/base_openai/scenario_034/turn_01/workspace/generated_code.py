
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
import pandas as pd
import numpy as np
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow analysis
andes.run(ssa)

# Define the voltage threshold
threshold = 1.030

# Count all buses above the threshold
above_threshold = ssa.Bus.voltage.v > threshold
selected_bus_ids = ssa.Bus.idx.v[above_threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
bus_voltages = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.voltage.v
})

lowest_buses = bus_voltages.nsmallest(2, 'voltage').sort_values(by='voltage')
lowest_bus_ids = lowest_buses['bus_id'].tolist()
lowest_voltages = lowest_buses['voltage'].round(6).tolist()

# Construct the result JSON
result_json = {
    'threshold': round(threshold, 6),
    'selected_bus_ids': selected_bus_ids,
    'selected_count': selected_count,
    'lowest_bus_ids': lowest_bus_ids,
    'lowest_voltages': lowest_voltages
}

# Print the result
print(f"RESULT_JSON={result_json}")