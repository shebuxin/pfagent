Here's the complete Python script that maintains the previously added load, scales every PQ load, reruns the power flow, and saves a line plot of bus voltage magnitude as specified:

```python
# required_dependencies: andes, matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_029_A'
new_load_bus = 4
p0 = 0.01
q0 = 0.006

ssa.PQ.add(new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Rerun the power flow
andes.run_power_flow(ssa)

# Scale every PQ load by a factor of 1.020
scale_factor = 1.020
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Rerun the power flow again
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.Bus.v.v

# Determine max and min bus voltages
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)
max_bus_index = np.where(bus_voltages == max_voltage)[0][0]
min_bus_index = np.where(bus_voltages == min_voltage)[0][0]

# Get corresponding bus IDs
max_bus = int(ssa.Bus.idx.v[max_bus_index])
min_bus = int(ssa.Bus.idx.v[min_bus_index])

# Save voltage magnitude plot
plt.figure()
plt.plot(bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.xticks(ticks=np.arange(len(bus_voltages)), labels=ssa.Bus.idx.v)
plt.grid()
plot_file = 'scenario_029_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare RESULT_JSON
RESULT_JSON = {
    "scale_factor": round(float(scale_factor), 6),
    "max_bus": max_bus,
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": min_bus,
    "min_voltage": round(float(min_voltage), 6),
    "plot_file": plot_file
}

# Output the result
print("RESULT_JSON=", RESULT_JSON)
```

This script incorporates all the specified requirements, from adding the PQ load and scaling it to running power flow and plotting the bus voltage magnitudes. Be sure to have the necessary libraries installed in your Python environment to run this script successfully.