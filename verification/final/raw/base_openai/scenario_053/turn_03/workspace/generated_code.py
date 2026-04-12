
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

# required_dependencies: pandas, numpy, matplotlib
import os
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_053.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.volt_target.set(first_pv_idx, 1.015)

# Scale every PQ load by 1.040
scale_factor = 1.040
ssa.PQ.p.set(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor)
ssa.PQ.q.set(ssa.PQ.idx.v, ssa.PQ.q.v * scale_factor)

# Rerun power flow
andes.run(ssa)

# Get the bus voltages
voltages = ssa.V.v
buses = ssa.bus.v

# Find min and max voltages
min_idx = np.argmin(voltages)
max_idx = np.argmax(voltages)

min_bus = int(buses[min_idx])
min_voltage = round(float(voltages[min_idx]), 6)
max_bus = int(buses[max_idx])
max_voltage = round(float(voltages[max_idx]), 6)

# Create a bar chart of the bus voltages
plt.bar(buses, voltages)
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages After Load Scaling')
plt.savefig('scenario_053_turn3_bar.png')
plt.close()

# Create the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": 'scenario_053_turn3_bar.png'
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")