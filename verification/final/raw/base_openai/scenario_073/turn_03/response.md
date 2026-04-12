Here’s the complete script that incorporates scaling of every PQ load, reruns the power flow, and saves a line plot of bus voltage magnitude to the specified file.

```python
# required_dependencies: andes, matplotlib
import andes
import os
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_073.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7
bus_7_idx = ssa.BUS.idx.v[ssa.BUS.id.v == 7]
if bus_7_idx.size == 0:
    raise ValueError("Bus 7 not found in the system.")
andes.PQ.add(
    ssa,
    idx="PQ_VERIFY_073_A",
    bus=bus_7_idx[0],
    p0=0.016,
    q0=0.01
)

# Run power flow
andes.run(ssa)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
for pq_idx in ssa.PQ.idx.v:
    p_current = ssa.PQ.p0.v[pq_idx]
    q_current = ssa.PQ.q0.v[pq_idx]
    ssa.PQ.p0.v[pq_idx] = round(p_current * scale_factor, 6)
    ssa.PQ.q0.v[pq_idx] = round(q_current * scale_factor, 6)

# Run power flow again
andes.run(ssa)

# Extract bus voltages for plotting
bus_ids = ssa.BUS.id.v
bus_voltages = ssa.BUS.vmag.v

# Plotting bus voltage magnitude
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plot_file = 'scenario_073_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Find max and min voltage buses
max_bus_idx = bus_voltages.argmax()
min_bus_idx = bus_voltages.argmin()
max_bus = bus_ids[max_bus_idx]
max_voltage = round(bus_voltages[max_bus_idx], 6)
min_bus = bus_ids[min_bus_idx]
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```

This script performs all the tasks specified, including scaling the PQ loads, rerunning the power flow, and generating the plot with the required output formatted in JSON.