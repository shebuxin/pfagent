
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

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 20
added_load_idx = "PQ_VERIFY_032_A"
ssa.PQ.add(bus=20, p0=0.019, q0=0.012, id=added_load_idx)

# Rerun power flow
andes.run_power_flow(ssa)

# Extract buses below 0.980 p.u. and the minimum voltage bus
threshold = 0.980
voltages = ssa.bus.Vmag.v
bus_ids = ssa.bus.idx.v

selected_bus_ids = [int(bus_ids[i]) for i in range(len(voltages)) if voltages[i] < threshold]
selected_count = len(selected_bus_ids)

# Finding the minimum voltage bus
min_bus_index = voltages.index(min(voltages))
min_bus = int(bus_ids[min_bus_index])
min_voltage = round(float(voltages[min_bus_index]), 6)

# Prepare Result JSON
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "added_load_bus": 20,
    "threshold": round(threshold, 6),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print("RESULT_JSON=", RESULT_JSON)