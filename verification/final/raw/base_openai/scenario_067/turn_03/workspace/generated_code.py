
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

# Load the Kundur built-in case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
pv_idx = ssa.PV.idx.v[0]
ssa.PV.set(pv_idx, target_voltage=1.010)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.set_loads(scale_factor=scale_factor)

# Rerun power flow
andes.run(ssa, log=False)

# Retrieve the bus voltage information
voltages = ssa.Bus.voltage.v
buses = ssa.Bus.idx.v

# Find min and max voltages
min_voltage = round(float(voltages.min()), 6)
max_voltage = round(float(voltages.max()), 6)
min_bus = int(buses[voltages.argmin()])
max_bus = int(buses[voltages.argmax()])

# Save a bar chart of the bus voltages
plt.bar(buses, voltages)
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages after Power Flow')
plt.grid(True)
plot_file = 'scenario_067_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Output results in the required JSON format
RESULT_JSON = {
    "scale_factor": scale_factor,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": plot_file
}

print(f"RESULT_JSON={RESULT_JSON}")