
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

# required_dependencies: andes, numpy
import andes
import numpy as np
import os

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_045.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get voltage information
bus_voltages = ssa.Bus.vm.v
bus_ids = ssa.Bus.idx.v

# Find the slack bus voltage and ID
slack_bus_idx = np.where(ssa.Bus.is_slack.v)[0][0]
slack_bus_id = bus_ids[slack_bus_idx]
slack_voltage = bus_voltages[slack_bus_idx]

# Get top 4 highest voltages (in descending order)
highest_voltage_indices = np.argsort(bus_voltages)[-4:][::-1]
selected_bus_ids = bus_ids[highest_voltage_indices].tolist()
selected_voltages = np.round(bus_voltages[highest_voltage_indices], 6).tolist()

# Prepare the result JSON
result_json = {
    "slack_bus": int(slack_bus_id),
    "slack_voltage": round(float(slack_voltage), 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result in the required format
print(f"RESULT_JSON={result_json}")