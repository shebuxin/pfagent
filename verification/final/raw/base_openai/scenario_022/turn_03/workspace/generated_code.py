
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
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_022.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.040
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v]
slack_setpoint = 1.040
ssa.Bus.v.set(slack_bus_idx, slack_setpoint)

# Add a new PQ load at bus 10
new_load_idx = 'PQ_VERIFY_022_B'
p0 = 0.018
q0 = 0.013
bus_number = 10

# Register new PQ load
ssa.PQ.add(bus=bus_number, idx=new_load_idx, p0=p0, q0=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Get max and min voltage buses
max_bus_idx = ssa.Bus.idx.v[np.argmax(ssa.Bus.v.v)]
min_bus_idx = ssa.Bus.idx.v[np.argmin(ssa.Bus.v.v)]
max_voltage = round(ssa.Bus.v.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v.v[min_bus_idx], 6)

# Count total PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus_idx),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus_idx),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result JSON
print("RESULT_JSON=" + json.dumps(RESULT_JSON))