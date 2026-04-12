
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

# Load the built-in IEEE 39 case
case_path = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
bus_index = 15
load_idx = 'PQ_VERIFY_042_D'
p0 = 0.018
q0 = 0.012

# Add the load
ssa.PQ.add(bus=bus_index, idx=load_idx, p0=p0, q0=q0)

# Setup completed, now run power flow
andes.run(ssa)

# Extract the slack bus information
slack_bus = ssa.Bus.idx.v[ssa.Bus.slack.v]
slack_voltage = ssa.Bus.Vmag.v[ssa.Bus.slack.v]

# Extract buses below 0.970 p.u.
threshold = 0.970
selected_bus_ids = [int(bus_id) for bus_id, voltage in zip(ssa.Bus.idx.v, ssa.Bus.Vmag.v) if voltage < threshold]
selected_count = len(selected_bus_ids)

# Print the result in the specified format
RESULT_JSON = {
    "added_load_idx": load_idx,
    "slack_bus": int(slack_bus),
    "slack_voltage": round(slack_voltage, 6),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}
print(f"RESULT_JSON={RESULT_JSON}")