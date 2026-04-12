
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

# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
added_load_idx = 'PQ_VERIFY_002_A'
p0 = 0.013
q0 = 0.008
bus_idx = ssa.Bus.idx.v.tolist().index(5)  # Find the index of bus 5
ssa.PQ.add(added_load_idx, bus=bus_idx, p0=p0, q0=q0)

# Run power flow again
andes.run_power_flow(ssa)

# Define the threshold
threshold = 1.005

# Find buses below the threshold
voltages = ssa.Bus.v
selected_bus_ids = [ssa.Bus.idx.v[i] for i in range(len(voltages)) if voltages[i] < threshold]

# Find the minimum voltage bus
min_bus_idx = voltages.index(min(voltages))
min_bus = ssa.Bus.idx.v[min_bus_idx]
min_voltage = round(voltages[min_bus_idx], 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "added_load_bus": 5,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")