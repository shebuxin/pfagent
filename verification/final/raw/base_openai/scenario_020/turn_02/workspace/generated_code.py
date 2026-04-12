
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
import os
import andes
import json
import numpy as np

# Load the case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_ieee14_020.xlsx")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Set slack bus voltage target
slack_setpoint = 1.025
slack_bus_idx = ssa.SlackBus.idx.v[0]  # Assuming there's only one slack bus
ssa.SlackBus.set(bus=slack_bus_idx, v_target=slack_setpoint)

# Run power flow
andes.run_power_flow(ssa)

# Get results
slack_voltage = ssa.Bus.v[slack_bus_idx]
bus_voltages = ssa.Bus.v

# Count how many buses fall below 1.005 p.u.
selected_count = np.sum(bus_voltages < 1.005)

# Prepare the result as required
result_json = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

# Print the result in the specified format
print("RESULT_JSON=" + json.dumps(result_json))