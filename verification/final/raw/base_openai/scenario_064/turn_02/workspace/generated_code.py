
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

# required_dependencies: numpy,pandas,json
import os
import andes
import numpy as np
import pandas as pd
import json

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[0]  # Assuming the first bus is the slack bus
slack_setpoint = 1.020
ssa.Bus.v.set(slack_setpoint, idx=slack_bus_idx)

# Run the power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.v.v[slack_bus_idx], 6)

# Count how many buses fall below 0.970 p.u.
buses_below_threshold = np.where(ssa.Bus.v.v < 0.970)[0]
selected_count = len(buses_below_threshold)

# Create the result JSON
result_json = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": slack_voltage,
    "selected_count": selected_count
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')