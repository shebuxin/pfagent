
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
case = os.path.join(script_dir, "verify_ieee39_054.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_054_D'
p0 = 0.014
q0 = 0.008
bus_4_idx = 4  # Assuming bus indexing starts at 1 for user input
ssa.PQ.add(idx=new_load_idx, bus=bus_4_idx, p0=p0, q0=q0)

# Prepare to run power flow
andes.run_power_flow(ssa)

# Get the slack bus specification
slack_bus_idx = ssa.Bus.slack.idx[0]
slack_voltage = round(ssa.Bus.vmag.v[slack_bus_idx - 1], 6)  # Adjust for 0-indexing

# Determine buses with voltage below 0.950 p.u.
threshold = 0.950
bus_ids = ssa.Bus.idx.v
bus_voltages = ssa.Bus.vmag.v

# Find buses below the voltage threshold
selected_buses = [(bus_ids[i], bus_voltages[i]) for i in range(len(bus_voltages)) if bus_voltages[i] < threshold]
selected_bus_ids = [bus[0] for bus in selected_buses]
selected_count = len(selected_bus_ids)

# Prepare the result json output
result_json = {
    "added_load_idx": new_load_idx,
    "slack_bus": slack_bus_idx,
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")