
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
import pandas as pd
import andes

# Load the built-in Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Define threshold
threshold = 1.010

# Count buses above the threshold
above_threshold = ssa.Bus.voltage.v > threshold
selected_bus_ids = ssa.Bus.idx.v[above_threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
voltages = ssa.Bus.voltage.v
lowest_bus_ids = ssa.Bus.idx.v[voltages.argsort()][:2].tolist()
lowest_voltages = [round(voltages[bus_id], 6) for bus_id in lowest_bus_ids]

# Constructing the final JSON-like output
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")