```python
# required_dependencies: andes,json,matplotlib
import andes
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_071.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by a factor of 1.030 after setup
scale_factor = 1.030
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Run power flow again
andes.run_power_flow(ssa)

# Retrieve bus voltages
voltage_data = ssa.Bus.voltage.v
bus_ids = ssa.Bus.id.v

# Find max and min voltage buses
max_voltage_idx = np.argmax(voltage_data)
min_voltage_idx = np.argmin(voltage_data)

max_bus = bus_ids[max_voltage_idx]
max_voltage = round(voltage_data[max_voltage_idx], 6)
min_bus = bus_ids[min_voltage_idx]
min_voltage = round(voltage_data[min_voltage_idx], 6)

# Save line plot of bus voltage magnitude
plot_file = 'scenario_071_turn3_line.png'
plt.figure()
plt.plot(bus_ids, voltage_data, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig(plot_file)
plt.close()

# Prepare the result
result_json = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "plot_file": plot_file
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")
```