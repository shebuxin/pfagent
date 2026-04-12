# required_dependencies: pandas, numpy, matplotlib
import os
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the case from the uploaded Excel file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
if ssa.PV.idx.v.size > 0:
    first_pv_index = ssa.PV.idx.v[0]
    ssa.PV.set(first_pv_index, "Vtarget", 1.010)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.set("P", ssa.PQ.P.v * scale_factor)
ssa.PQ.set("Q", ssa.PQ.Q.v * scale_factor)

# Rerun power flow
andes.run_power_flow(ssa)

# Retrieve bus voltages and their corresponding bus indices
voltages = ssa.bus.v

# Find min and max voltage buses
min_index = np.argmin(voltages)
max_index = np.argmax(voltages)

min_bus = int(ssa.bus.idx.v[min_index])
min_voltage = round(float(voltages[min_index]), 6)
max_bus = int(ssa.bus.idx.v[max_index])
max_voltage = round(float(voltages[max_index]), 6)

# Create a bar chart of bus voltages
plt.figure(figsize=(10, 6))
plt.bar(ssa.bus.idx.v, voltages, color='blue')
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.xticks(ssa.bus.idx.v)
plt.grid()
plt.savefig('scenario_081_turn3_bar.png')
plt.close()

# Prepare the RESULT_JSON
result_json = {
    "scale_factor": scale_factor,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": 'scenario_081_turn3_bar.png'
}

# Print the result in the specified format
print(f"RESULT_JSON={result_json}")