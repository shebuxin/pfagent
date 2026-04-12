
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

# Add a new PQ load at bus 9
load_idx = 'PQ_VERIFY_003_A'
p0 = 0.016
q0 = 0.01

# Find the index for bus 9
bus_9_idx = np.where(ssa.BUS.idx.v == 9)[0]
if len(bus_9_idx) == 0:
    raise ValueError("Bus 9 not found.")

# Add the PQ load
ssa.PQ.add(bus=bus_9_idx[0], idx=load_idx, p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Define threshold
threshold = 1.010

# Get all bus voltages and their indices
bus_voltages = ssa.BUS.v
bus_ids = ssa.BUS.idx.v

# Collect buses below the threshold
selected_bus_pairs = [(bus_ids[i], bus_voltages[i]) for i in range(len(bus_voltages)) if bus_voltages[i] < threshold]
selected_bus_ids = sorted([int(bus[0]) for bus in selected_bus_pairs])

# Find minimum voltage bus information
min_bus_idx = np.argmin(bus_voltages)
min_bus = int(bus_ids[min_bus_idx])
min_voltage = round(float(bus_voltages[min_bus_idx]), 6)

# Prepare RESULT_JSON
RESULT_JSON = {
    "added_load_idx": load_idx,
    "added_load_bus": 9,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")