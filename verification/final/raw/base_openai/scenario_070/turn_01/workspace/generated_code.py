
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
case_path = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Retrieve voltage data
voltages = ssa.Bus.v

# Find the 4 lowest-voltage buses
lowest_voltage_indices = voltages.argsort()[:4]
selected_bus_ids = ssa.Bus.idx.v[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in lowest_voltage_indices]

# Prepare the result JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)