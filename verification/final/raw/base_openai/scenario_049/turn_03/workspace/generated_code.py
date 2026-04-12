
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
case = os.path.join(script_dir, "verify_ieee39_049.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.030
slack_bus_idx = ssa.Bus.idx.v[0]  # Assuming the first bus is the slack bus
ssa.Bus.Vtarget.set(slack_bus_idx, 1.030)

# Add a new PQ load at bus 15
pq_load_idx = ssa.PQ.add(bus=15, idx='PQ_VERIFY_049_B', p0=0.016, q0=0.011)

# Rerun power flow
andes.run_power_flow(ssa)

# Determine max and min voltage buses
max_voltage = max(ssa.V.v)
min_voltage = min(ssa.V.v)
max_bus = ssa.Bus.idx.v[ssa.V.v.index(max_voltage)]
min_bus = ssa.Bus.idx.v[ssa.V.v.index(min_voltage)]

# Total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the results
result_json = {
    "added_load_idx": pq_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")