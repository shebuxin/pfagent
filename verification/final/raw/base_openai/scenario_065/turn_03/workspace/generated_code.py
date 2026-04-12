
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

# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the built-in Kundur case
case = andes.get_case("kundur/kundur_aw.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 0.990
ssa.PV.set(0, target_voltage=0.990)

# Scale every PQ load by 1.030
ssa.PQ.set_scaler(1.030)

# Rerun power flow
andes.run_powerflow(ssa)

# Get bus voltage information
voltages = ssa.Bus.v
min_voltage = round(float(voltages.min()), 6)
max_voltage = round(float(voltages.max()), 6)
min_bus = int(ssa.Bus.idx.v[voltages.argmin()])
max_bus = int(ssa.Bus.idx.v[voltages.argmax()])

# Save a bar chart of the bus voltages
plt.figure(figsize=(10, 6))
plt.bar(ssa.Bus.idx.v, voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Bus Voltage (p.u.)')
plt.title('Bus Voltages After Scaling PQ Loads')
plt.ylim(0.8, 1.2)
plt.grid()
plot_file = 'scenario_065_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result
result_json = {
    "scale_factor": 1.030,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": plot_file,
}

# Print result
print(f"RESULT_JSON={result_json}")