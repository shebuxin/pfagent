
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

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract bus voltages and their corresponding IDs
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v

# Combine bus ids and their voltages into a list of tuples
bus_voltage_pairs = list(zip(bus_ids, voltages))

# Sort by voltage
sorted_buses = sorted(bus_voltage_pairs, key=lambda x: x[1])

# Select the 4 lowest-voltage buses
lowest_buses = sorted_buses[:4]

# Separate the bus IDs and voltages
selected_bus_ids = [int(bus[0]) for bus in lowest_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_buses]

# Print the result in the required format
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}
print(f"RESULT_JSON={RESULT_JSON}")