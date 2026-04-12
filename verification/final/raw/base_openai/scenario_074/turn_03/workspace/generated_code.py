
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
import os
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_074.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9 (if not already added)
new_load_idx = 'PQ_VERIFY_074_A'
bus_9_idx = np.where(ssa.Bus.idx.v == 9)[0][0]  # Find the index for bus 9
if new_load_idx not in ssa.PQ.idx.v:
    ssa.PQ.add(new_load_idx, bus_9_idx, p0=0.019, q0=0.012)

# Scale every PQ load by a factor of 1.060 after setup
scale_factor = 1.060
for idx in range(len(ssa.PQ.idx.v)):
    ssa.PQ.p[idx] *= scale_factor
    ssa.PQ.q[idx] *= scale_factor

# Run power flow after scaling the loads
andes.run_powerflow(ssa)

# Extract voltage data
voltages = ssa.Bus.v.v

# Find max and min bus voltages
max_bus_idx = np.argmax(voltages)
min_bus_idx = np.argmin(voltages)
max_bus = int(ssa.Bus.idx.v[max_bus_idx])
max_voltage = round(float(voltages[max_bus_idx]), 6)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])
min_voltage = round(float(voltages[min_bus_idx]), 6)

# Save a line plot of bus voltage magnitude
plot_file = 'scenario_074_turn3_line.png'
plt.figure(figsize=(10, 5))
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)