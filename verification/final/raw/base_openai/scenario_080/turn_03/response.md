```python
# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the uploaded case
case = "verify_kundur_080.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
pv_index = ssa.PV.idx.v[0]  # Get the index of the first PV
ssa.PV.set(pv_index, 'Vtarget', 1.000)

# Scale every PQ load by 1.040
scale_factor = 1.040
ssa.PQ.set('P', ssa.PQ.p_mw * scale_factor)
ssa.PQ.set('Q', ssa.PQ.q_mvar * scale_factor)

# Rerun power flow
result = andes.run(ssa)

# Extract bus voltage information
bus_voltages = ssa.bus.v

# Find min and max voltage buses
min_bus_index = bus_voltages.argmin()
max_bus_index = bus_voltages.argmax()
min_voltage = round(bus_voltages.min(), 6)
max_voltage = round(bus_voltages.max(), 6)

# Prepare bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(bus_voltages)), bus_voltages, color='lightblue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages')
plt.xticks(range(len(bus_voltages)))
plt.grid(axis='y')

# Save the bar chart
plot_file = 'scenario_080_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare JSON result
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "min_bus": int(ssa.bus.idx.v[min_bus_index]),
    "min_voltage": float(min_voltage),
    "max_bus": int(ssa.bus.idx.v[max_bus_index]),
    "max_voltage": float(max_voltage),
    "plot_file": plot_file,
}

print("RESULT_JSON=", RESULT_JSON)
```