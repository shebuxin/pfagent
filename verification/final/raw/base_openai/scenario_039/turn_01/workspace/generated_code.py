
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
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Run power flow analysis
case.solve_power_flow()

# Extract voltage information
voltages = case.Bus.v
buses = case.Bus.idx

max_voltage_idx = voltages.argmax()
min_voltage_idx = voltages.argmin()

max_bus = int(buses[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)
min_bus = int(buses[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Create the JSON result
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")