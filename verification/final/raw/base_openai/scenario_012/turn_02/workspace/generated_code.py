
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

# Load the IEEE 14 full case
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
load_idx = case.PQ.add(bus=4, p0=0.014, q0=0.008, name='PQ_VERIFY_012_D')

# Rerun the power flow
case.run()

# Get the slack bus and its voltage
slack_bus = case.Bus.idx.v[case.Slack.idx.v[0]]
slack_voltage = round(float(case.Bus.v.v[case.Slack.idx.v[0]]), 6)

# Get all bus voltages and find those below 1.000 p.u.
voltages = case.Bus.v.v
threshold = 1.0
selected_bus_ids = [int(case.Bus.idx.v[i]) for i in range(len(voltages)) if voltages[i] < threshold]
selected_count = len(selected_bus_ids)

# Create the result JSON object
RESULT_JSON = {
    "added_load_idx": int(load_idx),
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")