```python
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
```