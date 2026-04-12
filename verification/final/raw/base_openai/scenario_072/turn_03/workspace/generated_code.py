
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

# required_dependencies: numpy,andes,matplotlib
import numpy as np
import andes
import os
import matplotlib.pyplot as plt

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_072.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add PQ load at bus 6
new_load_idx = 'PQ_VERIFY_072_A'
new_load_bus = 6
p0 = 0.013
q0 = 0.008
ssa.PQ.add(idx=new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Scale every PQ load by a factor of 1.040
scale_factor = 1.040
for idx in ssa.PQ.idx.v:
    ssa.PQ.set(idx=idx, p0=ssa.PQ.p0.v[ssa.PQ.idx.v.tolist().index(idx)] * scale_factor,
                q0=ssa.PQ.q0.v[ssa.PQ.idx.v.tolist().index(idx)] * scale_factor)

# Rerun power flow
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.vmag.v
bus_ids = ssa.Bus.idx.v

# Find max and min voltage buses
max_bus_idx = np.argmax(bus_voltages)
min_bus_idx = np.argmin(bus_voltages)
max_bus = int(bus_ids[max_bus_idx])
max_voltage = round(bus_voltages[max_bus_idx], 6)
min_bus = int(bus_ids[min_bus_idx])
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Save line plot of bus voltage magnitude
plot_file = 'scenario_072_turn3_line.png'
plt.figure(figsize=(10, 6))
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plt.savefig(plot_file)
plt.close()

# Prepare result
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

print("RESULT_JSON=", RESULT_JSON)