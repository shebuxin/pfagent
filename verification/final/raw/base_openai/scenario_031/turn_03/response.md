```python
# required_dependencies: andes,json,matplotlib
import andes
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
bus_number = 15
load_idx = 'PQ_VERIFY_031_A'
p0 = 0.016
q0 = 0.01
ssa.PQ.add(bus=bus_number, idx=load_idx, p0=p0, q0=q0)

# Setup case after adding the load
andes.setup(ssa)

# Scale every PQ load by a factor of 1.040
scale_factor = 1.040
for idx in ssa.PQ.idx.v:
    ssa.PQ.set(idx=idx, p0=ssa.PQ.p0.v[idx] * scale_factor, q0=ssa.PQ.q0.v[idx] * scale_factor)

# Run power flow
andes.run_power_flow(ssa)

# Find max and min bus voltages
bus_voltages = ssa.Bus.Vmag.v
max_bus_idx = np.argmax(bus_voltages)
min_bus_idx = np.argmin(bus_voltages)
max_bus = int(ssa.Bus.idx.v[max_bus_idx])
max_voltage = round(bus_voltages[max_bus_idx], 6)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Save a line plot of bus voltage magnitude
plt.figure()
plt.plot(ssa.Bus.idx.v, bus_voltages, marker='o')
plt.xlabel('Bus Index')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('Bus Voltage Magnitudes After Scaling Loads')
plt.xticks(ssa.Bus.idx.v)
plt.grid()
plot_file = 'scenario_031_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Construct the result JSON
result_json = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file,
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))
```