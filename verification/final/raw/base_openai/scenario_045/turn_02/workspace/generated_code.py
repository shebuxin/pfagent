
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

# required_dependencies: andes, numpy
import andes
import numpy as np
import os

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_045.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
new_load_idx = 'PQ_VERIFY_045_A'
new_load_bus = 15
p0 = 0.016
q0 = 0.01

ssa.PQ.add(idx=new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Run power flow again
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.vm.v
bus_ids = ssa.Bus.idx.v

# Identify buses below the threshold of 0.970 p.u.
threshold = 0.970
selected_bus_ids = bus_ids[bus_voltages < threshold]
selected_bus_ids.sort()  # Sort bus IDs in ascending order
selected_count = len(selected_bus_ids)

# Find the minimum voltage bus
min_voltage_index = np.argmin(bus_voltages)
min_bus = int(bus_ids[min_voltage_index])
min_voltage = round(float(bus_voltages[min_voltage_index]), 6)

# Prepare the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "added_load_bus": int(new_load_bus),
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids.tolist(),
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result in the required format
print(f"RESULT_JSON={result_json}")