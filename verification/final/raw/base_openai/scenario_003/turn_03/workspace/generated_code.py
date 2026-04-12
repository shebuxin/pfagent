
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

# Add a new PQ load at bus 9
load_idx = 'PQ_VERIFY_003_A'
p0 = 0.016
q0 = 0.01

# Find the index for bus 9
bus_9_idx = np.where(ssa.BUS.idx.v == 9)[0]
if len(bus_9_idx) == 0:
    raise ValueError("Bus 9 not found.")

# Add the PQ load
ssa.PQ.add(bus=bus_9_idx[0], idx=load_idx, p0=p0, q0=q0)

# Run initial power flow
andes.run_power_flow(ssa)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
ssa.PQ.P.v *= scale_factor
ssa.PQ.Q.v *= scale_factor

# Rerun power flow
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.BUS.v
bus_ids = ssa.BUS.idx.v

# Find max and min voltage buses
max_bus_idx = np.argmax(bus_voltages)
min_bus_idx = np.argmin(bus_voltages)
max_bus = int(bus_ids[max_bus_idx])
max_voltage = round(float(bus_voltages[max_bus_idx]), 6)
min_bus = int(bus_ids[min_bus_idx])
min_voltage = round(float(bus_voltages[min_bus_idx]), 6)

# Save a line plot of bus voltage magnitudes
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_003_turn3_line.png')
plt.close()

# Prepare RESULT_JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": 'scenario_003_turn3_line.png'
}

print(f"RESULT_JSON={RESULT_JSON}")