
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

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_043.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
bus_4_idx = 4  # Assuming bus 4 is correctly indexed
new_load_idx = 'PQ_VERIFY_043_A'
new_p0 = 0.01
new_q0 = 0.006
ssa.PQ.add(idx=new_load_idx, bus=bus_4_idx, p0=new_p0, q0=new_q0)

# Setup after adding the load
ssa = ssa.setup()

# Run power flow
andes.run(ssa)

# Parameters for analysis
threshold = 0.950
bus_voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Find buses below the threshold and the minimum voltage bus
selected_bus_ids = []
min_bus_idx = np.argmin(bus_voltages)
min_voltage = float(bus_voltages[min_bus_idx])
min_bus = int(ssa.Bus.idx[min_bus_idx])

for idx, voltage in zip(bus_ids, bus_voltages):
    if float(voltage) < threshold:
        selected_bus_ids.append(int(idx))

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": bus_4_idx,
    "threshold": round(threshold, 6),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")