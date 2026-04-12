
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

# Load the Kundur full case
case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6
load_idx = case.PQ.idx.add()  # Create a new load index
case.PQ.idx.v[load_idx] = 'PQ_VERIFY_058_A'
case.PQ.bus.v[load_idx] = 6
case.PQ.p0.v[load_idx] = 0.013
case.PQ.q0.v[load_idx] = 0.008

# Run power flow after adding the load
andes.run_powerflow(case)

# Set threshold and gather data
threshold = 0.950
selected_bus_ids = []
min_voltage = float('inf')
min_bus = None

for bus_id in range(len(case.Bus.v)):
    voltage = case.Bus.v[bus_id]
    if voltage < threshold:
        selected_bus_ids.append(int(bus_id))
    if voltage < min_voltage:
        min_voltage = voltage
        min_bus = int(bus_id)

# Count of selected buses below threshold
selected_count = len(selected_bus_ids)

# Build the result JSON object
RESULT_JSON = {
    "added_load_idx": int(load_idx),
    "added_load_bus": 6,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6),
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")