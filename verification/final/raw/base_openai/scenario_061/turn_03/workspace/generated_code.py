
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
import andes
import os
import json

# Load the Kundur full case
script_dir = os.getcwd()
case = os.path.join(script_dir, "kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack bus voltage target
slack_bus_idx = ssa.Bus.idx.v[np.where(ssa.Bus.type.v == 'slack')[0][0]]  # Get index of slack bus
slack_setpoint = 0.990
ssa.V.set(slack_bus_idx, slack_setpoint)

# Add new PQ load at bus 4
new_load_idx = ssa.PQ.add(bus=4, name='PQ_VERIFY_061_B', p0=0.012, q0=0.007)

# Rerun power flow
andes.run_power_flow(ssa)

# Analyze max and min voltage buses
voltages = ssa.Bus.v.val
max_voltage = voltages.max()
min_voltage = voltages.min()
max_bus = ssa.Bus.idx.v[np.argmax(voltages)]
min_bus = ssa.Bus.idx.v[np.argmin(voltages)]

# Count total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Create JSON result
result_json = {
    "added_load_idx": int(new_load_idx),
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))