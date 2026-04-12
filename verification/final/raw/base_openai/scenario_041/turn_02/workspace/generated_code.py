
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
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load before setup
pq_load_idx = ssa.PQ.add(bus=10, idx='PQ_VERIFY_041_D', p0=0.016, q0=0.01)

# Re-run power flow
andes.run_power_flow(ssa)

# Retrieve the slack bus voltage and buses below 0.960 p.u.
slack_bus = ssa.Bus.idx[ssa.slack.idx].v[0]
slack_voltage = ssa.Bus.v[ssa.slack.idx].v[0]
threshold = 0.960
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v < threshold].tolist()
selected_count = len(selected_bus_ids)

# Prepare the result
RESULT_JSON = {
    "added_load_idx": pq_load_idx,
    "slack_bus": slack_bus,
    "slack_voltage": round(slack_voltage, 6),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result in the specified JSON format
print("RESULT_JSON=", RESULT_JSON)