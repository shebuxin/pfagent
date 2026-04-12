
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
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_077.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.type.v == 3][0]  # Assuming type 3 is the slack bus
slack_setpoint = 1.010
ssa.Bus.v.target.v[slack_bus] = slack_setpoint

# Add a new PQ load at bus 7
new_pq_idx = 'PQ_VERIFY_077_B'
p0 = 0.016
q0 = 0.011
bus_number = 7
andes.create_PQ(ssa, new_pq_idx, bus_number, p0, q0)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get the maximum and minimum voltage buses
max_bus = ssa.Bus.idx.v[ssa.Bus.v.v.argmax()]
max_voltage = round(ssa.Bus.v.v.max(), 6)
min_bus = ssa.Bus.idx.v[ssa.Bus.v.v.argmin()]
min_voltage = round(ssa.Bus.v.v.min(), 6)
total_pq_count = ssa.PQ.idx.v.size

# Form the result
RESULT_JSON = {
    "added_load_idx": new_pq_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": int(total_pq_count)
}

print(f"RESULT_JSON={RESULT_JSON}")