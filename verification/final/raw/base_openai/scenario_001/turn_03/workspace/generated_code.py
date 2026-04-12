
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
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
new_load_idx = ssa.PQ.add('PQ_VERIFY_001_A', p0=0.01, q0=0.006, bus=4)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Run power flow
andes.run_power_flow(ssa)

# Get all bus voltages
voltages = ssa.bus.v

# Identify max and min bus voltages
max_bus = int(np.argmax(voltages))
max_voltage = round(float(voltages[max_bus]), 6)
min_bus = int(np.argmin(voltages))
min_voltage = round(float(voltages[min_bus]), 6)

# Save line plot of bus voltage magnitude
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(voltages)), voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_001_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
result_json = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")