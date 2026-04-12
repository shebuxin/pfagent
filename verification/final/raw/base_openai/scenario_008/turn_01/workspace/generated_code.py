
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

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Run power flow
andes.run(ssa, no_output=True)

# Define the threshold
threshold = 1.030

# Count buses above threshold and find their IDs
bus_ids = ssa.Bus.idx.v
bus_voltages = ssa.Bus.v.v
selected_bus_ids = [int(bus_id) for bus_id, voltage in zip(bus_ids, bus_voltages) if voltage > threshold]
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_bus_ids = sorted(zip(bus_ids, bus_voltages), key=lambda x: x[1])[:2]
lowest_bus_ids = [int(bus_id) for bus_id, voltage in lowest_bus_ids]
lowest_voltages = [round(voltage, 6) for bus_id, voltage in lowest_bus_ids]

# Prepare the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")