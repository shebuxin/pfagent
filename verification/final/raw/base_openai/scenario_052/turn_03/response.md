```python
# required_dependencies: numpy,pandas,andes,matplotlib
import numpy as np
import pandas as pd
import os
import andes
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_052.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Modify the first PV's voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.v_target.set(first_pv_idx, 1.010)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.set(ssa.PQ.idx.v, ssa.PQ.p.v * scale_factor)
ssa.PQ.q.set(ssa.PQ.idx.v, ssa.PQ.q.v * scale_factor)

# Run power flow analysis again
result = andes.run_power_flow(ssa)

# Prepare voltage data for plotting
voltages = ssa.Bus.v.v
bus_indices = np.arange(len(voltages))

# Find minimum and maximum voltage buses
min_idx = np.argmin(voltages)
max_idx = np.argmax(voltages)

min_bus = ssa.Bus.idx.v[min_idx]
max_bus = ssa.Bus.idx.v[max_idx]
min_voltage = round(float(voltages[min_idx]), 6)
max_voltage = round(float(voltages[max_idx]), 6)

# Create a bar chart of the bus voltages
plt.figure(figsize=(10, 6))
plt.bar(bus_indices, voltages, color='blue')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages After Scaling Loads')
plt.grid()
plt.tight_layout()
plot_file = 'scenario_052_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "plot_file": plot_file
}

print(f"RESULT_JSON={RESULT_JSON}")
```