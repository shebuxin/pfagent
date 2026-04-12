
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
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_074.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_074_A'
bus_9_idx = np.where(ssa.Bus.idx.v == 9)[0][0]  # Find the index for bus 9
ssa.PQ.add(new_load_idx, bus_9_idx, p0=0.019, q0=0.012)

# Run power flow after adding the load
andes.run_powerflow(ssa)

# Extract voltage data
voltages = ssa.Bus.v.v

# Find buses below 0.970 p.u. and the minimum-voltage bus
threshold = 0.970
selected_bus_ids = ssa.Bus.idx.v[voltages < threshold].tolist()
min_bus_idx = np.argmin(voltages)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])
min_voltage = round(float(voltages[min_bus_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": 9,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)