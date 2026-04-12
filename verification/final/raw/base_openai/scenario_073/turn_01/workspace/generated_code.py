
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
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_073.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Extracting slack bus information
slack_bus_idx = ssa.BUS.idx.v[ssa.BUS.slack.v]
slack_voltage = ssa.BUS.vmag.v[slack_bus_idx]

# Getting all bus voltages and their IDs
bus_ids = ssa.BUS.id.v
bus_voltages = ssa.BUS.vmag.v

# Pairing bus IDs with voltages and sorting to get top 4 highest voltages
sorted_buses = sorted(zip(bus_ids, bus_voltages), key=lambda x: x[1], reverse=True)
top_buses = sorted_buses[:4]

# Preparing the output format
selected_bus_ids = [bus[0] for bus in top_buses]
selected_voltages = [round(bus[1], 6) for bus in top_buses]

# Result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")