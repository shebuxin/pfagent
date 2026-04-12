
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

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_100.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract voltage data
slack_bus = ssa.Slack.idx.v[0]  # Slack bus index
slack_voltage = round(ssa.Slack.V.v[0], 6)  # Slack bus voltage

# Maximum and minimum voltage bus calculations
max_voltage = round(ssa.Bus.V.v.max(), 6)
min_voltage = round(ssa.Bus.V.v.min(), 6)

max_bus = ssa.Bus.idx.v[ssa.Bus.V.v.argmax()]  # Bus index with max voltage
min_bus = ssa.Bus.idx.v[ssa.Bus.V.v.argmin()]  # Bus index with min voltage

# Prepare results in JSON format
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)