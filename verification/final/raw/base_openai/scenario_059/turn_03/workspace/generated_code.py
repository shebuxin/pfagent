
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
import numpy as np
import matplotlib.pyplot as plt

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True)

# Add a new PQ load at bus 7
bus_7_idx = np.where(ssa.Bus.idx.v == 7)[0][0]
ssa.PQ.add(idx='PQ_VERIFY_059_A', bus=bus_7_idx, p0=0.016, q0=0.01)

# Setup the modified case
andes.setup(ssa)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Run power flow again
andes.solve()

# Extract the bus data for plotting
voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find maximum and minimum bus voltages
max_voltage = round(np.max(voltages), 6)
max_bus = int(ssa.Bus.idx.v[np.argmax(voltages)])
min_voltage = round(np.min(voltages), 6)
min_bus = int(ssa.Bus.idx.v[np.argmin(voltages)])

# Save the line plot of bus voltage magnitude
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_059_turn3_line.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": 'scenario_059_turn3_line.png'
}

print(f"RESULT_JSON={RESULT_JSON}")