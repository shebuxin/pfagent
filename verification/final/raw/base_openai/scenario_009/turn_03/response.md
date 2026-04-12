```python
# required_dependencies: andes, matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.Vtarget.set(first_pv_idx, 1.010)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.P.set(ssa.PQ.idx.v, ssa.PQ.P.v * scale_factor)
ssa.PQ.Q.set(ssa.PQ.idx.v, ssa.PQ.Q.v * scale_factor)

# Rerun power flow
andes.run(ssa)

# Get bus voltages and their statistics
bus_voltages = ssa.Bus.Vm.v
min_voltage = round(float(np.min(bus_voltages)), 6)
max_voltage = round(float(np.max(bus_voltages)), 6)
min_bus = int(np.where(bus_voltages == np.min(bus_voltages))[0][0]) + 1  # Convert to 1-based index
max_bus = int(np.where(bus_voltages == np.max(bus_voltages))[0][0]) + 1  # Convert to 1-based index

# Save bar chart of bus voltages
plt.bar(range(1, len(bus_voltages) + 1), bus_voltages)
plt.title('Bus Voltages')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.ylim(0.9, 1.1)  # Set y-limits for better visualization
plt.grid()
plot_file = 'scenario_009_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the JSON result
RESULT_JSON = {
    "scale_factor": scale_factor,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```