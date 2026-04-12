
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

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_075.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the slack bus ID
slack_setpoint = 0.990
ssa.Bus.vmag.set(slack_setpoint, index=ssa.Bus.idx.v == slack_bus)

# Add a new PQ load at bus 4
pq_load_idx = 'PQ_VERIFY_075_B'
p0 = 0.012
q0 = 0.007
new_pq_idx = ssa.PQ.add(bus=4, idx=pq_load_idx, p=p0, q=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Get voltage details
max_voltage = ssa.Bus.vmag.v.max()
min_voltage = ssa.Bus.vmag.v.min()
max_bus = ssa.Bus.idx.v[ssa.Bus.vmag.v == max_voltage].item()
min_bus = ssa.Bus.idx.v[ssa.Bus.vmag.v == min_voltage].item()
total_pq_count = ssa.PQ.idx.v.size

# Prepare result in JSON format
result_json = {
    "added_load_idx": pq_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": int(total_pq_count)
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")