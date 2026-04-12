
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
import numpy as np
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_083.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6
bus_idx = np.where(ssa.Bus.idx.v == 6)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 6 not found in the system.")
sssa.PQ.add(idx='PQ_VERIFY_083_D', bus=6, p0=0.016, q0=0.01)

# Run power flow
andes.run(ssa)

# Obtain the slack bus and its voltage
slack_idx = ssa.Slack.idx.v[0]
slack_voltage = round(ssa.Bus.V.v[slack_idx], 6)

# Identify buses below 0.950 p.u.
threshold = 0.950
selected_bus_indices = np.where(ssa.Bus.V.v < threshold)[0]
selected_bus_ids = ssa.Bus.idx.v[selected_bus_indices].tolist()
selected_count = len(selected_bus_ids)

# Prepare result in JSON format
result_json = {
    "added_load_idx": 'PQ_VERIFY_083_D',
    "slack_bus": slack_idx,
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

print(f"RESULT_JSON={json.dumps(result_json)}")