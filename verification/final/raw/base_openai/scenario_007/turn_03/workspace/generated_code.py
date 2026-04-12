
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
import andes
import numpy as np
import pandas as pd

# Load the IEEE 14 full case
ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Slack.idx.v[0]
slack_setpoint = 1.035
ssa.Slack.v_target.set(slack_setpoint)

# Add a new PQ load at bus 9
added_load_idx = "PQ_VERIFY_007_B"
p0 = 0.016
q0 = 0.011
bus_number = 9
ssa.PQ.add(bus=bus_number, idx=added_load_idx, p0=p0, q0=q0)

# Run power flow analysis
andes.run(ssa)

# Get the max and min voltage buses and total PQ loads count
voltage_levels = ssa.Bus.v
max_bus = np.argmax(voltage_levels)
min_bus = np.argmin(voltage_levels)
max_voltage = round(voltage_levels[max_bus], 6)
min_voltage = round(voltage_levels[min_bus], 6)
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the JSON result
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print(f"RESULT_JSON={RESULT_JSON}")