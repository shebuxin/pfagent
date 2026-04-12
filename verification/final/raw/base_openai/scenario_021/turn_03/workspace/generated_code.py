
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
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_021.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack bus voltage target
slack_bus_idx = ssa.Bus.idx[0]  # Assuming the first bus is the slack bus
slack_setpoint = 1.035
ssa.V.set(slack_bus_idx, slack_setpoint)

# Add a new PQ load at bus 9
pq_load_idx = 'PQ_VERIFY_021_B'
p0 = 0.016
q0 = 0.011
bus_9_idx = ssa.Bus.idx[8]  # Assuming bus 9 is the 9th bus in idx

ssa.PQ.add(pq_load_idx, bus=bus_9_idx, p=p0, q=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Find the maximum and minimum voltage buses
max_bus_idx = ssa.Bus.idx[np.argmax(ssa.Bus.v)]
min_bus_idx = ssa.Bus.idx[np.argmin(ssa.Bus.v)]
max_voltage = round(ssa.Bus.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v[min_bus_idx], 6)

# Count total number of PQ loads
total_pq_count = len(ssa.PQ.idx)

# Prepare the result
result_json = {
    "added_load_idx": pq_load_idx,
    "max_bus": max_bus_idx,
    "max_voltage": max_voltage,
    "min_bus": min_bus_idx,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print(f"RESULT_JSON={json.dumps(result_json)}")