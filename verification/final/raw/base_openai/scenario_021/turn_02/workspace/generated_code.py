
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
case = os.path.join(script_dir, "verify_ieee14_021.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack bus voltage target
slack_bus_idx = ssa.Bus.idx[0]  # Assuming the first bus is the slack bus
slack_setpoint = 1.035
ssa.V.set(slack_bus_idx, slack_setpoint)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = ssa.Bus.v[slack_bus_idx]

# Count the buses below 1.010 p.u.
below_threshold_buses = ssa.Bus.ss[ssa.Bus.v < 1.010]
selected_count = len(below_threshold_buses)

# Prepare the result
result_json = {
    "slack_bus": slack_bus_idx,
    "slack_setpoint": slack_setpoint,
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

print(f"RESULT_JSON={json.dumps(result_json)}")