
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

# required_dependencies: numpy,andes
import os
import numpy as np
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_046.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.power_flow(ssa)

# Get slack bus voltage
slack_bus_idx = ssa.Slack.idx.v[0]
slack_bus_voltage = ssa.Bus.v[slack_bus_idx]

# Get all bus voltages and their indices
bus_voltages = ssa.Bus.v
bus_ids = np.arange(len(bus_voltages))

# Get top 5 highest voltage buses
top_indices = np.argsort(-bus_voltages)[:5]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = np.round(bus_voltages[top_indices].tolist(), 6)

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_bus_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")