
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

# required_dependencies: pandas,json,andes
import pandas as pd
import json
import andes

case = "verify_ieee14_016.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
new_load_idx = 'PQ_VERIFY_016_A'
p0 = 0.013
q0 = 0.008
bus_number = 5

# Get actual bus index for the new load
bus_idx = np.where(ssa.Bus.idx.v == bus_number)[0]
if bus_idx.size == 0:
    raise ValueError(f"Bus {bus_number} not found.")
bus_idx = bus_idx[0]

# Set the new load
ssa.PQ.add(new_load_idx, p0, q0, bus_idx)

# Run power flow after adding the load
andes.run_power_flow()

# Find buses below the voltage threshold
threshold = 1.005
voltages = ssa.Bus.V.v
selected_bus_ids = ssa.Bus.idx.v[voltages < threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the minimum voltage bus
min_voltage_idx = np.argmin(voltages)
min_bus = ssa.Bus.idx.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Prepare the result
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": bus_number,
    "threshold": float(threshold),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": int(selected_count),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")