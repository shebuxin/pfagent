
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

# required_dependencies: numpy,pandas
import os
import andes
import numpy as np
import pandas as pd

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_018.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
load_idx = 'PQ_VERIFY_018_A'
bus_number = 10
p0 = 0.019
q0 = 0.012
bus_mask = np.where(ssa.Bus.idx.v == bus_number)[0]

if bus_mask.size == 0:
    raise ValueError(f"No bus found with number {bus_number}")

# Adding the PQ load
ssa.PQ.set(load_idx, bus=bus_mask[0], p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Set threshold
threshold = 1.015

# Gather bus voltage data
bus_voltages = ssa.Bus.Vmag.v
bus_ids = ssa.Bus.idx.v

# Find buses below the threshold
below_threshold_mask = bus_voltages < threshold
selected_bus_ids = bus_ids[below_threshold_mask].tolist()
selected_bus_ids.sort()  # sort in ascending order

# Find the bus with minimum voltage
min_index = np.argmin(bus_voltages)
min_bus = bus_ids[min_index]
min_voltage = round(bus_voltages[min_index], 6)

# Prepare result JSON
RESULT_JSON = {
    "added_load_idx": load_idx,
    "added_load_bus": bus_number,
    "threshold": float(threshold),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)