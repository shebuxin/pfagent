
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
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Inspect bus voltages
bus_voltages = ssa.Bus.Vm.v
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)

# Get corresponding bus indices
max_bus = int(np.where(bus_voltages == max_voltage)[0][0]) + 1  # Convert to 1-based index
min_bus = int(np.where(bus_voltages == min_voltage)[0][0]) + 1  # Convert to 1-based index

# Prepare the JSON result
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": min_bus,
    "min_voltage": round(float(min_voltage), 6)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")