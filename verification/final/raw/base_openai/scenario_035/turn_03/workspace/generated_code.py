
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

# Load the case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
idx_new_load = ssa.PQ.add(bus=15, name='PQ_VERIFY_035_B', p0=0.016, q0=0.011)

# Set the slack-bus voltage target to 1.030
slack_bus_index = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
ssa.Vref.v[slack_bus_index] = 1.030

# Rerun power flow
andes.run_power_flow(ssa)

# Find maximum and minimum voltage buses
max_bus_index = np.argmax(ssa.Bus.v.v)
min_bus_index = np.argmin(ssa.Bus.v.v)

# Prepare the JSON output
RESULT_JSON = {
    "added_load_idx": idx_new_load,
    "max_bus": ssa.Bus.idx.v[max_bus_index],
    "max_voltage": round(ssa.Bus.v.v[max_bus_index], 6),
    "min_bus": ssa.Bus.idx.v[min_bus_index],
    "min_voltage": round(ssa.Bus.v.v[min_bus_index], 6),
    "total_pq_count": len(ssa.PQ.idx.v)
}

print("RESULT_JSON=", RESULT_JSON)