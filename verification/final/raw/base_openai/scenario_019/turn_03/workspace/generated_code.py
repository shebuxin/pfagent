
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

# required_dependencies: numpy,pandas,andes
import os
import numpy as np
import pandas as pd
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_019.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the slack bus index
ssa.Bus.v.set(slack_bus_idx, 1.020)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_019_B'
ssa.PQ.add(idx=new_load_idx, bus=4, p0=0.012, q0=0.007)

# Rerun power flow
andes.run_power_flow(ssa)

# Determine max-voltage and min-voltage bus
max_voltage = round(np.max(ssa.Bus.v.v), 6)
min_voltage = round(np.min(ssa.Bus.v.v), 6)
max_bus = ssa.Bus.idx.v[np.argmax(ssa.Bus.v.v)]
min_bus = ssa.Bus.idx.v[np.argmin(ssa.Bus.v.v)]

# Count the total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print("RESULT_JSON=", RESULT_JSON)