
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

# required_dependencies: andes
import andes
import numpy as np
import json

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_index = ssa.bus.slack.v[0]  # Getting the index of the slack bus
ssa.PV.set(slack_bus_index, v_set=1.035)

# Add a new PQ load at bus 20
pq_load_idx = ssa.PQ.add(bus=20, name='PQ_VERIFY_036_B', p0=0.018, q0=0.013)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the maximum and minimum voltage bus information
max_voltage_index = np.argmax(ssa.Bus.v)
min_voltage_index = np.argmin(ssa.Bus.v)
max_bus = int(max_voltage_index)
max_voltage = round(ssa.Bus.v[max_voltage_index], 6)
min_bus = int(min_voltage_index)
min_voltage = round(ssa.Bus.v[min_voltage_index], 6)

# Count total PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the JSON result
result_json = {
    "added_load_idx": int(pq_load_idx),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result in the specified format
print(f"RESULT_JSON={json.dumps(result_json)}")