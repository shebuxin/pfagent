
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

# required_dependencies: pandas, matplotlib
import andes
import os
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.005
pv_idx = ssa.PV.idx.v[0]
ssa.PV.vset.v[pv_idx] = 1.005

# Scale every PQ load by 1.020
scale_factor = 1.020
ssa.PQ.P.v *= scale_factor
ssa.PQ.Q.v *= scale_factor

# Rerun the power flow analysis
andes.powerflow(ssa)

# Extract bus voltages
voltages = ssa.Bus.v.v

# Find max and min voltage buses
max_voltage = max(voltages)
min_voltage = min(voltages)
max_bus = ssa.Bus.idx.v[voltages.argmax()]
min_bus = ssa.Bus.idx.v[voltages.argmin()]

# Save a bar chart of the bus voltages
plt.figure(figsize=(10, 6))
plt.bar(ssa.Bus.idx.v, voltages)
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages After Power Flow')
plt.grid()
plt.savefig('scenario_037_turn3_bar.png')
plt.close()

# Prepare the JSON output
RESULT_JSON = {
    "scale_factor": round(float(scale_factor), 6),
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6),
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "plot_file": 'scenario_037_turn3_bar.png'
}

# Print the result
print("RESULT_JSON=" + str(RESULT_JSON))