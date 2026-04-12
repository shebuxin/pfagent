
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
import andes
import os
import json

# Load the Kundur full case
script_dir = os.getcwd()
case = os.path.join(script_dir, "kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Analyze bus voltages
voltages = ssa.Bus.v.val
buses = ssa.Bus.idx.v

# Threshold for bus voltage
threshold = 0.990

# Find buses above threshold
selected_bus_ids = list(buses[voltages > threshold])
selected_count = len(selected_bus_ids)

# Find the two lowest voltage buses
lowest_indices = voltages.argsort()[:2]
lowest_bus_ids = buses[lowest_indices].tolist()
lowest_voltages = [round(voltages[i], 6) for i in lowest_indices]

# Create JSON result
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))