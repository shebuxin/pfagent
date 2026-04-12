```python
# required_dependencies: numpy,andes,matplotlib
import numpy as np
import andes
import matplotlib.pyplot as plt

# Load the case
case = "verify_pjm5_096.json"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale all PQ loads by 1.060
scale_factor = 1.060
ssa.PQ.set(ssa.PQ.p.v * scale_factor, ssa.PQ.q.v * scale_factor)

# Add new PQ load at bus 4 with idx 'PQ_VERIFY_096_E'
new_load_idx = 'PQ_VERIFY_096_E'
ssa.PQ.add(bus=4, p0=0.018, q0=0.012, idx=new_load_idx)

# Run power flow
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v

# Find max and min voltages and corresponding bus indices
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)
max_bus = np.where(bus_voltages == max_voltage)[0][0] + 1  # 1-based index
min_bus = np.where(bus_voltages == min_voltage)[0][0] + 1  # 1-based index

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_voltages)
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_096_turn3_voltage.png')

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6),
    "plot_file": 'scenario_096_turn3_voltage.png'
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```