```python
# required_dependencies: numpy,pandas,matplotlib
import os
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_015.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

# Add new PQ load at bus 4
bus_idx = np.where(ssa.bus.idx.v == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 does not exist in the case.")
bus_idx = bus_idx[0]

ssa.PQ.add(name='PQ_VERIFY_015_A', bus=bus_idx, p0=0.01, q0=0.006)

# Setup the case
ssa.setup()

# Run power flow
andes.powerflow(ssa)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
ssa.PQ.set({'p0': ssa.PQ.p0.v * scale_factor, 'q0': ssa.PQ.q0.v * scale_factor})

# Rerun power flow
andes.powerflow(ssa)

# Get bus voltages
voltages = ssa.bus.v
bus_ids = ssa.bus.idx.v

# Find max and min voltage buses
max_index = np.argmax(voltages)
min_index = np.argmin(voltages)

max_bus = int(bus_ids[max_index])
max_voltage = round(float(voltages[max_index]), 6)
min_bus = int(bus_ids[min_index])
min_voltage = round(float(voltages[min_index]), 6)

# Save a line plot of bus voltage magnitude
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.axhline(y=1.0, color='r', linestyle='--', label='1.0 p.u. Threshold')
plt.legend()
plt.grid()
plot_file = 'scenario_015_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare JSON output
result_json = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={result_json}")
```