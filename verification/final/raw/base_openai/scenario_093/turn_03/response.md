```python
# required_dependencies: andes, matplotlib
import andes
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_pjm5_093.json")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Add a new PQ load at bus 1 with idx 'PQ_VERIFY_093_E'
new_idx = 'PQ_VERIFY_093_E'
bus_number = 1
p0 = 0.012
q0 = 0.006
ssa.PQ.insert(new_idx, bus_number, p0, q0)

# Run power flow
andes.run_power_flow(ssa, log=False)

# Get bus voltage data
voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v
max_voltage = np.max(voltages)
min_voltage = np.min(voltages)
max_bus = bus_ids[np.argmax(voltages)]
min_bus = bus_ids[np.argmin(voltages)]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_093_turn3_voltage.png'
plt.savefig(plot_file)
plt.close()

# Prepare the RESULT_JSON
RESULT_JSON = {
    "added_load_idx": new_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": plot_file
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```