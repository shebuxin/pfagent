
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

# required_dependencies: numpy, pandas, json
import os
import numpy as np
import pandas as pd
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[np.where(ssa.Bus.type.v == 'slack')[0][0]]  # Assuming only one slack bus
slack_setpoint = 1.035
ssa.VM.set(slack_bus_idx, slack_setpoint)

# Add new PQ load at bus 20
pq_load_idx = ssa.PQ.add(p0=0.018, q0=0.013, bus=20, idx='PQ_VERIFY_050_B')

# Rerun power flow
andes.run(ssa)

# Get maximum and minimum voltage buses
max_bus_idx = np.argmax(ssa.Bus.v.v)
min_bus_idx = np.argmin(ssa.Bus.v.v)
max_voltage = round(ssa.Bus.v.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v.v[min_bus_idx], 6)

# Total count of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Constructing the result JSON
result_json = {
    "added_load_idx": pq_load_idx,
    "max_bus": max_bus_idx,
    "max_voltage": max_voltage,
    "min_bus": min_bus_idx,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count,
}

# Printing the result
print("RESULT_JSON=" + json.dumps(result_json))