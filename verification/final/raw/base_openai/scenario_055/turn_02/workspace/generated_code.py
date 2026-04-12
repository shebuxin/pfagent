
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

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_055.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
new_load_idx = 'PQ_VERIFY_055_D'
p0 = 0.016
q0 = 0.01
ssa.PQ.add(bus=10, idx=new_load_idx, p0=p0, q0=q0)

# Prepare for power flow analysis
andes.setup(ssa)

# Run power flow analysis
andes.run_power_flow()

# Get Slack Bus information
slack_bus_idx = ssa.Bus.slack.v[0]
slack_voltage = round(float(ssa.Bus.vmag.v[slack_bus_idx]), 6)

# Get buses with voltage below 0.960 p.u.
threshold = 0.960
selected_bus_ids = [int(ssa.Bus.idx.v[i]) for i in range(len(ssa.Bus.vmag.v)) if ssa.Bus.vmag.v[i] < threshold]
selected_count = len(selected_bus_ids)

# Prepare the result
RESULT_JSON = json.dumps({
    "added_load_idx": new_load_idx,
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
})

# Print the result as JSON
print(f"RESULT_JSON={RESULT_JSON}")