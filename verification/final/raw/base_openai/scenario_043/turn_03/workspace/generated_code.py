
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
case = os.path.join(script_dir, "verify_ieee39_043.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
bus_4_idx = 4
new_load_idx = 'PQ_VERIFY_043_A'
new_p0 = 0.01
new_q0 = 0.006
ssa.PQ.add(idx=new_load_idx, bus=bus_4_idx, p0=new_p0, q0=new_q0)

# Setup after adding the load
ssa = ssa.setup()

# Scale every PQ load by a factor of 1.020
scale_factor = 1.020
ssa.PQ.p0 *= scale_factor
ssa.PQ.q0 *= scale_factor

# Run power flow
andes.run(ssa)

# Get voltage information
bus_voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Find maximum and minimum voltage buses
max_bus_idx = np.argmax(bus_voltages)
min_bus_idx = np.argmin(bus_voltages)
max_voltage = round(float(bus_voltages[max_bus_idx]), 6)
min_voltage = round(float(bus_voltages[min_bus_idx]), 6)
max_bus = int(ssa.Bus.idx[max_bus_idx])
min_bus = int(ssa.Bus.idx[min_bus_idx])

# Save the line plot of bus voltage magnitude
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.tight_layout()
plot_file = 'scenario_043_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")