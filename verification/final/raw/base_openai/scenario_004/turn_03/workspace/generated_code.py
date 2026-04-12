
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

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load
ssa.PQ.add(
    idx='PQ_VERIFY_004_A',
    bus=10,
    p0=0.019,
    q0=0.012
)

# Run power flow analysis
andes.fpn(ssa)

# Scale every PQ load by a factor of 1.060
scale_factor = 1.060
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Run power flow analysis again
andes.fpn(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.V.v

# Find maximum and minimum voltage and corresponding buses
max_voltage = bus_voltages.max()
min_voltage = bus_voltages.min()
max_bus = int(ssa.Bus.idx.v[bus_voltages.argmax()])
min_bus = int(ssa.Bus.idx.v[bus_voltages.argmin()])

# Plot bus voltage magnitude
plt.figure()
plt.plot(ssa.Bus.idx.v, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_004_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the RESULT_JSON
RESULT_JSON = {
    'scale_factor': round(scale_factor, 6),
    'max_bus': max_bus,
    'max_voltage': round(max_voltage, 6),
    'min_bus': min_bus,
    'min_voltage': round(min_voltage, 6),
    'plot_file': plot_file
}

# Print the result in the specified format
print(f"RESULT_JSON={RESULT_JSON}")