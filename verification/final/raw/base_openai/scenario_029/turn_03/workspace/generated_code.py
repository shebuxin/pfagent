
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
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_029_A'
new_load_bus = 4
p0 = 0.01
q0 = 0.006

ssa.PQ.add(new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Rerun the power flow
andes.run_power_flow(ssa)

# Scale every PQ load by a factor of 1.020
scale_factor = 1.020
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Rerun the power flow again
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v

# Determine max and min bus voltages
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)
max_bus_index = np.where(bus_voltages == max_voltage)[0][0]
min_bus_index = np.where(bus_voltages == min_voltage)[0][0]

# Get corresponding bus IDs
max_bus = int(ssa.Bus.idx.v[max_bus_index])
min_bus = int(ssa.Bus.idx.v[min_bus_index])

# Save voltage magnitude plot
plt.figure()
plt.plot(bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.xticks(ticks=np.arange(len(bus_voltages)), labels=ssa.Bus.idx.v)
plt.grid()
plot_file = 'scenario_029_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare RESULT_JSON
RESULT_JSON = {
    "scale_factor": round(float(scale_factor), 6),
    "max_bus": max_bus,
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": min_bus,
    "min_voltage": round(float(min_voltage), 6),
    "plot_file": plot_file
}

# Output the result
print("RESULT_JSON=", RESULT_JSON)