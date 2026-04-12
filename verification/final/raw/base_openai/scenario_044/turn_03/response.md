```python
# required_dependencies: numpy,pandas,matplotlib
import os
import andes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_044.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Rerun power flow
andes.run_powerflow(ssa)

# Find max and min voltages and buses
voltages = ssa.Bus.Vm.v
max_voltage_idx = np.argmax(voltages)
min_voltage_idx = np.argmin(voltages)

max_bus = ssa.Bus.id.v[max_voltage_idx]
max_voltage = round(voltages[max_voltage_idx], 6)
min_bus = ssa.Bus.id.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Save a line plot of bus voltage magnitudes
plot_file = 'scenario_044_turn3_line.png'
plt.figure()
plt.plot(ssa.Bus.id.v, voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plt.savefig(plot_file)
plt.close()

# Prepare JSON result
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "plot_file": plot_file,
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")
```