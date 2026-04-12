
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

# required_dependencies: numpy,json
import os
import numpy as np
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_023.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v
bus_numbers = ssa.Bus.idx.v

# Determine max and min voltages and their corresponding buses
max_voltage = float(np.max(bus_voltages))
min_voltage = float(np.min(bus_voltages))
max_bus = int(bus_numbers[np.argmax(bus_voltages)])
min_bus = int(bus_numbers[np.argmin(bus_voltages)])

# Prepare the RESULT_JSON output
result_json = {
    "max_bus": max_bus,
    "max_voltage": round(max_voltage, 6),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6)
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')