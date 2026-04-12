
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

# required_dependencies: numpy,json,matplotlib
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_023.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.v_target.set_value(first_pv_idx, 1.010)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.set_value(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor)
ssa.PQ.q.set_value(ssa.PQ.idx.v, ssa.PQ.q.v * scale_factor)

# Rerun power flow
andes.power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v
bus_numbers = ssa.Bus.idx.v

# Determine max and min voltages and their corresponding buses
min_voltage = float(np.min(bus_voltages))
max_voltage = float(np.max(bus_voltages))
min_bus = int(bus_numbers[np.argmin(bus_voltages)])
max_bus = int(bus_numbers[np.argmax(bus_voltages)])

# Create a bar chart of the bus voltages
plt.figure(figsize=(10, 6))
plt.bar(bus_numbers, bus_voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages (p.u.)')
plt.xticks(bus_numbers)
plt.grid()
plot_file = 'scenario_023_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the RESULT_JSON output
result_json = {
    "scale_factor": round(scale_factor, 6),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6),
    "max_bus": max_bus,
    "max_voltage": round(max_voltage, 6),
    "plot_file": plot_file
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')