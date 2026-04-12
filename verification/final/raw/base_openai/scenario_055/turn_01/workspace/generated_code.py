
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
import json

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_055.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow()

# Get the bus voltages and sort them
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v
sorted_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])
lowest_voltage_buses = sorted_indices[:3]

# Prepare the result
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_buses]
selected_voltages = [round(float(voltages[i]), 6) for i in lowest_voltage_buses]

# Print the result as JSON
RESULT_JSON = json.dumps({
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
})
print(f"RESULT_JSON={RESULT_JSON}")