
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

# Load the case
case = os.path.join(os.getcwd(), "verify_pjm5_097.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow()

# Get voltage info
slack_bus = ssa.bus.idx.where(ssa.bus.slack.v)
slack_voltage = ssa.bus.v[slack_bus].round(6)

max_bus = ssa.bus.idx.where(ssa.bus.v == ssa.bus.v.max())
max_voltage = ssa.bus.v[max_bus].round(6)

min_bus = ssa.bus.idx.where(ssa.bus.v == ssa.bus.v.min())
min_voltage = ssa.bus.v[min_bus].round(6)

# Construct the RESULT_JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")