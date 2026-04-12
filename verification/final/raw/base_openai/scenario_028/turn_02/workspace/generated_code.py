
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
import os
import andes
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_028.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_028_D'
ssa.PQ.add(bus=9, idx=new_load_idx, p0=0.018, q0=0.012)

# Run power flow
andes.run_power_flow(ssa)

# Get the slack bus information
slack_bus = ssa.Bustype.idx[0]  # Assuming the first bus is the slack bus
slack_voltage = round(ssa.Bus.v[slack_bus], 6)

# Find buses below 1.010 p.u.
threshold = 1.010
selected_bus_indices = [i for i, v in enumerate(ssa.Bus.v) if v < threshold]
selected_bus_ids = ssa.Bus.idx[selected_bus_indices].tolist()
selected_count = len(selected_bus_ids)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")