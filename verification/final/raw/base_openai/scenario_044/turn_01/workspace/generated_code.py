
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

# required_dependencies: numpy,pandas
import os
import andes
import numpy as np
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_044.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Find the slack bus and its voltage
slack_bus_idx = np.where(ssa.Bus.slack.v)[0][0]
slack_bus = ssa.Bus.id.v[slack_bus_idx]
slack_voltage = round(ssa.Bus.Vm.v[slack_bus_idx], 6)

# Get top 3 highest voltage buses
voltages = ssa.Bus.Vm.v
top_indices = np.argsort(voltages)[-3:][::-1]  # indices of top-3 highest voltages
selected_bus_ids = ssa.Bus.id.v[top_indices].tolist()
selected_voltages = np.round(voltages[top_indices], 6).tolist()

# Prepare JSON result
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")