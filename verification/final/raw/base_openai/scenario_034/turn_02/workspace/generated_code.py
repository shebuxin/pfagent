
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

# required_dependencies: pandas, numpy
import pandas as pd
import numpy as np
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack bus voltage target
slack_setpoint = 1.020
slack_bus_idx = ssa.Bus.slack.v[0]  # Assuming the first slack bus is the relevant one

# Update the slack bus voltage setpoint
ssa.Bus.voltage_target[slack_bus_idx] = slack_setpoint

# Run the power flow analysis again
andes.run(ssa)

# Retrieve the slack bus voltage
slack_voltage = ssa.Bus.voltage.v[slack_bus_idx]

# Count buses falling below 0.960 p.u.
below_threshold = ssa.Bus.voltage.v < 0.960
selected_count = np.sum(below_threshold)

# Construct the result JSON
result_json = {
    'slack_bus': int(slack_bus_idx),
    'slack_setpoint': round(slack_setpoint, 6),
    'slack_voltage': round(slack_voltage, 6),
    'selected_count': selected_count
}

# Print the result
print(f"RESULT_JSON={result_json}")