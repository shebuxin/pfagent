
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
import pandas as pd
import andes
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
bus_idx = np.where(ssa.Bus.idx.v == 10)[0][0]
ssa.PQ.add(idx='PQ_VERIFY_030_A', bus=bus_idx, p0=0.013, q0=0.008)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
ssa.PQ.p0.v = ssa.PQ.p0.v * scale_factor
ssa.PQ.q0.v = ssa.PQ.q0.v * scale_factor

# Rerun power flow
andes.run_power_flow(ssa)

# Get bus voltages
voltages = ssa.Bus.voltage.v
bus_ids = ssa.Bus.idx.v

# Determine max and min bus voltages
max_voltage_idx = np.argmax(voltages)
min_voltage_idx = np.argmin(voltages)

max_bus = int(bus_ids[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)
min_bus = int(bus_ids[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)

# Create a line plot of bus voltage magnitudes
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_030_turn3_line.png'
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