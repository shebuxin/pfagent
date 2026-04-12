
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

# required_dependencies: andes,json,matplotlib
import andes
import json
import os
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_094.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale each PQ load by 1.040
scale_factor = 1.040
ssa.PQ.set(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor, ssa.PQ.q.v * scale_factor)

# Add a new PQ load at bus 2
added_load_idx = 'PQ_VERIFY_094_E'
p0 = 0.014
q0 = 0.008
ssa.PQ.add(bus=2, idx=added_load_idx, p=p0, q=q0)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get bus voltage data
bus_ids = ssa.Bus.idx.v
bus_voltages = ssa.Bus.v.v  # Assuming 'v' contains voltage values

# Find the max and min voltage buses
max_voltage = max(bus_voltages)
min_voltage = min(bus_voltages)
max_bus = bus_ids[bus_voltages.index(max_voltage)]
min_bus = bus_ids[bus_voltages.index(min_voltage)]

# Plot bus voltages
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (pu)')
plt.grid()
plt.savefig('scenario_094_turn3_voltage.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "max_bus": max_bus,
    "max_voltage": round(max_voltage, 6),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_094_turn3_voltage.png'
}

# Print the result
print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")