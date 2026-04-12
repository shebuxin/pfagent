
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

# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_048.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_setpoint = 1.020
slack_bus = ssa.Slack.idx.v[0]  # Assuming there's at least one slack bus
ssa.Slack.v_setpoint.set(slack_setpoint)

# Add new PQ load at bus 10
load_idx = 'PQ_VERIFY_048_B'
p0 = 0.014
q0 = 0.009
bus_id = 10
ssa.PQ.add(idx=load_idx, bus=bus_id, p0=p0, q0=q0)

# Rerun power flow
andes.run_pf(ssa)

# Find maximum and minimum voltage bus
voltages = ssa.Bus.v  # Get all bus voltages
max_bus = np.argmax(voltages)  # Index of maximum voltage bus
min_bus = np.argmin(voltages)  # Index of minimum voltage bus
max_voltage = voltages[max_bus]  # Maximum voltage
min_voltage = voltages[min_bus]  # Minimum voltage

# Count total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "added_load_idx": load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")