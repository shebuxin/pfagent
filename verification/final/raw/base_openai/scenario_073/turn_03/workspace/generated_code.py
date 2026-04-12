
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
import os
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_073.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7
bus_7_idx = ssa.BUS.idx.v[ssa.BUS.id.v == 7]
if bus_7_idx.size == 0:
    raise ValueError("Bus 7 not found in the system.")
andes.PQ.add(
    ssa,
    idx="PQ_VERIFY_073_A",
    bus=bus_7_idx[0],
    p0=0.016,
    q0=0.01
)

# Run power flow
andes.run(ssa)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
for pq_idx in ssa.PQ.idx.v:
    p_current = ssa.PQ.p0.v[pq_idx]
    q_current = ssa.PQ.q0.v[pq_idx]
    ssa.PQ.p0.v[pq_idx] = round(p_current * scale_factor, 6)
    ssa.PQ.q0.v[pq_idx] = round(q_current * scale_factor, 6)

# Run power flow again
andes.run(ssa)

# Extract bus voltages for plotting
bus_ids = ssa.BUS.id.v
bus_voltages = ssa.BUS.vmag.v

# Plotting bus voltage magnitude
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plot_file = 'scenario_073_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Find max and min voltage buses
max_bus_idx = bus_voltages.argmax()
min_bus_idx = bus_voltages.argmin()
max_bus = bus_ids[max_bus_idx]
max_voltage = round(bus_voltages[max_bus_idx], 6)
min_bus = bus_ids[min_bus_idx]
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")