
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
import matplotlib.pyplot as plt

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.060
load_scale_factor = 1.060
ssa.PQ.P.v *= load_scale_factor
ssa.PQ.Q.v *= load_scale_factor

# Add a new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_088_E'
ssa.PQ.add(idx=new_load_idx, bus=4, p0=0.018, q0=0.012)

# Run power flow
andes.run(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.V.v
bus_indices = ssa.Bus.idx.v

# Find maximum and minimum voltage buses
max_voltage = max(bus_voltages)
min_voltage = min(bus_voltages)
max_bus = bus_indices[bus_voltages.index(max_voltage)]
min_bus = bus_indices[bus_voltages.index(min_voltage)]

# Save line plot of bus voltages
plt.plot(bus_indices, bus_voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_088_turn3_voltage.png')
plt.close()

# Print the final JSON result
RESULT_JSON = json.dumps({
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_088_turn3_voltage.png'
})
print(f"RESULT_JSON={RESULT_JSON}")