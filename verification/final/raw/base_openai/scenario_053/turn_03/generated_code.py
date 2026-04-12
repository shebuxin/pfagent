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