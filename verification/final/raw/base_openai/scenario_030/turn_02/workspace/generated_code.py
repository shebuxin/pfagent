
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
import pandas as pd
import andes
import numpy as np
import os

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
bus_idx = np.where(ssa.Bus.idx.v == 10)[0][0]
ssa.PQ.add(idx='PQ_VERIFY_030_A', bus=bus_idx, p0=0.013, q0=0.008)

# Rerun power flow
andes.run_power_flow(ssa)

# Get bus voltages
voltages = ssa.Bus.voltage.v
bus_ids = ssa.Bus.idx.v

# Define threshold for minimum voltage
threshold = 0.960

# Identify buses below threshold and find the minimum voltage bus
selected_bus_indices = np.where(voltages < threshold)[0]
selected_bus_ids = list(bus_ids[selected_bus_indices])
selected_count = len(selected_bus_ids)
min_voltage_idx = np.argmin(voltages)
min_bus = int(bus_ids[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": "PQ_VERIFY_030_A",
    "added_load_bus": 10,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)