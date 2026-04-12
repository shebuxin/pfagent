
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
case = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True)

# Set the slack-bus voltage target
slack_setpoint = 1.025
case.Slack.set("Vtarget", slack_setpoint)

# Add a new PQ load before setup at bus 5
added_load_idx = case.PQ.add(bus=5, name='PQ_VERIFY_006_B', p0=0.014, q0=0.009)

# Rerun power flow
case.run()

# Report max and min voltage buses
voltages = case.Bus.v
max_bus = voltages.argmax()  # Bus with maximum voltage
min_bus = voltages.argmin()  # Bus with minimum voltage

# Prepare the response data
max_voltage = round(voltages[max_bus], 6)
min_voltage = round(voltages[min_bus], 6)
total_pq_count = len(case.PQ.idx.v)

# Prepare the result in required JSON format
RESULT_JSON = json.dumps({
    "added_load_idx": added_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
})

print(f"RESULT_JSON={RESULT_JSON}")