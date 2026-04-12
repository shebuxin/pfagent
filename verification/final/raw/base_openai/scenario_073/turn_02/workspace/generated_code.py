
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
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_073.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7
bus_7_idx = ssa.BUS.idx.v[ssa.BUS.id.v == 7]
if bus_7_idx.size == 0:
    raise ValueError("Bus 7 not found in the system.")
andes.PQ.add(
    ssa,
    idx="PQ_VERIFY_073_A",
    bus=bus_7_idx[0],
    p0=0.016,
    q0=0.01
)

# Run power flow
andes.run(ssa)

# Get buses below 0.960 p.u. and the minimum voltage bus
threshold = 0.960
bus_ids = ssa.BUS.id.v
bus_voltages = ssa.BUS.vmag.v

# Find buses below the threshold
selected_buses = [(bus_id, voltage) for bus_id, voltage in zip(bus_ids, bus_voltages) if voltage < threshold]
selected_buses.sort(key=lambda x: x[0])  # Sort by bus ID

# Prepare to find the minimum voltage bus
min_bus_idx = bus_voltages.argmin()
min_bus = bus_ids[min_bus_idx]
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Prepare the output format
added_load_idx = "PQ_VERIFY_073_A"
added_load_bus = 7
selected_bus_ids = [bus[0] for bus in selected_buses]
selected_count = len(selected_bus_ids)

# Result JSON
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "added_load_bus": added_load_bus,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")