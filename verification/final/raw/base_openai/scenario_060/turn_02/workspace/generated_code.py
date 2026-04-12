
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

# required_dependencies: andes,json
import andes
import json

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_060_A'
p0 = 0.019
q0 = 0.012
bus_9_idx = np.where(ssa.PQ.bus.v == 9)[0][0]
ssa.PQ.add(new_load_idx, bus_9_idx, p0, q0)

# Setup is required before running power flow
andes.setup(ssa)

# Run power flow
andes.run_power_flow(ssa)

# Identify buses below 0.970 p.u.
threshold = 0.970
bus_voltages = ssa.PQ.v.v
bus_ids = ssa.PQ.bus.v

selected_bus_ids = [int(bus_ids[i]) for i in range(len(bus_voltages)) if bus_voltages[i] < threshold]
selected_bus_ids.sort()  # Sort the selected bus ids

# Find the minimum voltage bus
min_voltage_idx = np.argmin(bus_voltages)
min_bus = int(bus_ids[min_voltage_idx])
min_voltage = round(bus_voltages[min_voltage_idx], 6)

# Prepare the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "added_load_bus": 9,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))