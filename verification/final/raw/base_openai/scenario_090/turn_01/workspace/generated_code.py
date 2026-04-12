
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

# Load the built-in PJM 5-bus case 
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Extract voltage results
voltage = ssa.Bus.V.v
slack_idx = ssa.Bus.idx.slack
slack_voltage = round(voltage[slack_idx], 6)

# Identify max and min voltage buses
max_voltage_idx = voltage.argmax()
min_voltage_idx = voltage.argmin()

max_bus = ssa.Bus.idx.v[max_voltage_idx]
min_bus = ssa.Bus.idx.v[min_voltage_idx]
max_voltage = round(voltage[max_voltage_idx], 6)
min_voltage = round(voltage[min_voltage_idx], 6)

# Prepare the result
result = {
    "slack_bus": int(max_bus),
    "slack_voltage": slack_voltage,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

# Print the result in the required format
print(f"RESULT_JSON={result}")