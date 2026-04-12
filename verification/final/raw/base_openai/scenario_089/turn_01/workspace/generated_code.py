
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

# required_dependencies: andes, json
import andes
import json

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Run power flow
case.solve()

# Get voltage information
voltages = case.PQ.v
buses = case.PQ.bus.v

# Find max and min voltage
max_idx = voltages.argmax()
min_idx = voltages.argmin()
slack_idx = case.Slack.idx.v[0]  # Assuming there is only one slack bus

# Prepare the RESULT_JSON
RESULT_JSON = {
    "slack_bus": int(slack_idx),
    "slack_voltage": round(case.PQ.v[slack_idx], 6),
    "max_bus": int(buses[max_idx]),
    "max_voltage": round(voltages[max_idx], 6),
    "min_bus": int(buses[min_idx]),
    "min_voltage": round(voltages[min_idx], 6)
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")