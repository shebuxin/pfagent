
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

# required_dependencies: pandas, numpy, andes
import andes
import pandas as pd
import numpy as np

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the index of the slack bus
ssa.V.set(target=1.020, idx=[slack_bus])

# Add a new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_005_B'
bus_number = 4
p0 = 0.012
q0 = 0.007
ssa.PQ.add(idx=new_load_idx, bus=bus_number, p0=p0, q0=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Get voltage information
voltages = ssa.Bus.v[:]
buses = np.arange(len(voltages)) + 1  # Bus IDs are 1-based

# Determine max and min voltage buses
max_bus = buses[np.argmax(voltages)]
max_voltage = round(voltages.max(), 6)
min_bus = buses[np.argmin(voltages)]
min_voltage = round(voltages.min(), 6)

# Count total PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Construct result JSON object
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": int(total_pq_count)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")