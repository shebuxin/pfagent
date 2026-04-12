
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
import os
import json
import andes
import numpy as np

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_076.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Slack.idx.v[0]  # Assuming there is one Slack bus
slack_setpoint = 1.0
ssa.Slack.vset.v[0] = slack_setpoint

# Add a new PQ load at bus 6
new_load_idx = 'PQ_VERIFY_076_B'
p0 = 0.014
q0 = 0.009
andes.PQ.add(ssa, idx=new_load_idx, bus=6, p=p0, q=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Find the maximum and minimum voltage buses
max_bus_idx = np.argmax(ssa.Bus.v)
min_bus_idx = np.argmin(ssa.Bus.v)
max_bus = ssa.Bus.idx.v[max_bus_idx]
min_bus = ssa.Bus.idx.v[min_bus_idx]
max_voltage = round(ssa.Bus.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v[min_bus_idx], 6)

# Count the total number of PQ loads present
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result as JSON
result_json = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")