
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

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7
pq_load_idx = 'PQ_VERIFY_070_D'
bus_7_idx = 7  # Bus 7 index
p0 = 0.018
q0 = 0.012

# Add the PQ load
ssa.PQ.add(buses=bus_7_idx, p0=p0, q0=q0, idx=pq_load_idx)

# Setup modified case
ssa.setup()

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage
slack_bus_idx = ssa.bus.v[ssa.Bus.slack_flag.v].idx.v[0]
slack_voltage = round(ssa.Bus.v[slack_bus_idx], 6)

# Find buses with voltage below 0.960 p.u.
threshold = 0.960
selected_bus_indices = [i for i, v in enumerate(ssa.Bus.v) if v < threshold]
selected_bus_ids = ssa.Bus.idx.v[selected_bus_indices].tolist()
selected_count = len(selected_bus_ids)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": pq_load_idx,
    "slack_bus": slack_bus_idx,
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)