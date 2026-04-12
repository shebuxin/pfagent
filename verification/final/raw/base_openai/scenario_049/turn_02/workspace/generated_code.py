
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
case = os.path.join(script_dir, "verify_ieee39_049.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.030
slack_bus_idx = ssa.Bus.idx.v[0]  # Assuming the first bus is the slack bus
ssa.Bus.Vtarget.set(slack_bus_idx, 1.030)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage and count buses below 0.970 p.u.
slack_voltage = ssa.V.v[slack_bus_idx]
selected_count = sum(voltage < 0.970 for voltage in ssa.V.v)

# Prepare the results
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_setpoint": round(1.030, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")