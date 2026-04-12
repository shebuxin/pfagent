
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

# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.slack.v[0]
slack_setpoint = 1.040
ssa.Bus.v.target[slack_bus_idx] = slack_setpoint

# Run power flow
andes.run(ssa, no_output=True)

# Report slack bus voltage
slack_voltage = ssa.Bus.v.v[slack_bus_idx]

# Count buses below 1.015 p.u.
bus_voltages = ssa.Bus.v.v
selected_count = sum(1 for voltage in bus_voltages if voltage < 1.015)

# Prepare the result JSON
result_json = {
    "slack_bus": int(slack_bus_idx),
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_count": selected_count
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")