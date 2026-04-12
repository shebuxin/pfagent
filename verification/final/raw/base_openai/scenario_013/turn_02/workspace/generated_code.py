
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
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
new_load = ssa.PQ.add(bus=5, idx='PQ_VERIFY_013_D', p0=0.016, q0=0.01)

# Set up the system again after adding the load
andes.setup(ssa)

# Run power flow analysis
andes.run_power_flow(ssa)

# Get the slack bus and its voltage
slack_bus = ssa.Bus.slack_idx.v[0]
slack_voltage = ssa.Bus.V.v[slack_bus]

# Identify buses with voltages below the threshold
threshold = 1.005
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v
selected_bus_ids = bus_ids[voltages < threshold].tolist()
selected_count = len(selected_bus_ids)

# Construct the result JSON
RESULT_JSON = {
    "added_load_idx": new_load.idx,
    "slack_bus": slack_bus,
    "slack_voltage": round(slack_voltage, 6),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}
print("RESULT_JSON=", RESULT_JSON)