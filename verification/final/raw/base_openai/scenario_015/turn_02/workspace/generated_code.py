
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

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_015.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

# Add new PQ load at bus 4
bus_idx = np.where(ssa.bus.idx.v == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 does not exist in the case.")
bus_idx = bus_idx[0]

ssa.PQ.add(name='PQ_VERIFY_015_A', bus=bus_idx, p0=0.01, q0=0.006)

# Setup the case
ssa.setup()

# Run power flow
andes.powerflow(ssa)

# Get bus voltages
voltages = ssa.bus.v
bus_ids = ssa.bus.idx.v

# Threshold for reporting
threshold = 1.0

# Find buses below the threshold
selected_bus_ids = bus_ids[voltages < threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the minimum voltage bus
min_index = np.argmin(voltages)
min_bus = int(bus_ids[min_index])
min_voltage = round(float(voltages[min_index]), 6)

# Prepare JSON output
result_json = {
    "added_load_idx": 'PQ_VERIFY_015_A',
    "added_load_bus": int(bus_idx),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={result_json}")