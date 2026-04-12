
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
ssa.Bus.voltage_target[slack_bus_idx] = slack_setpoint

# Add a new PQ load at bus 10
new_load_idx = 'PQ_VERIFY_034_B'
p0 = 0.014
q0 = 0.009
ssa.PQ.add(bus=10, p0=p0, q0=q0, idx=new_load_idx)

# Run the power flow analysis again
andes.run(ssa)

# Retrieve the maximum and minimum voltage buses
max_bus_idx = ssa.Bus.idx.v[np.argmax(ssa.Bus.voltage.v)]
max_voltage = round(np.max(ssa.Bus.voltage.v), 6)
min_bus_idx = ssa.Bus.idx.v[np.argmin(ssa.Bus.voltage.v)]
min_voltage = round(np.min(ssa.Bus.voltage.v), 6)

# Count the total number of PQ loads present
total_pq_count = ssa.PQ.idx.v.size

# Construct the result JSON
result_json = {
    'added_load_idx': new_load_idx,
    'max_bus': int(max_bus_idx),
    'max_voltage': max_voltage,
    'min_bus': int(min_bus_idx),
    'min_voltage': min_voltage,
    'total_pq_count': total_pq_count
}

# Print the result
print(f"RESULT_JSON={result_json}")