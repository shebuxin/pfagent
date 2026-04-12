
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

# Load the Kundur built-in case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa, log=False)

# Retrieve bus voltage values
voltages = ssa.Bus.voltage.v
buses = ssa.Bus.idx.v

# Find max and min voltages and their corresponding bus indices
max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

max_bus = int(buses[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)
min_bus = int(buses[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Output results in the required JSON format
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")