Here's the complete runnable Python script that adjusts the PV voltage target, scales all PQ loads by 1.030, reruns the power flow, and saves a bar chart of the bus voltages as specified:

```python
# required_dependencies: andes, matplotlib
import andes
import os
import json
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_079.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Set the first PV voltage target to 0.990
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.set('Vtarget', [0.990], idx=[first_pv_idx])

# Scale all PQ loads by 1.030
scale_factor = 1.030
ssa.PQ.set('P', ssa.PQ.P * scale_factor)
ssa.PQ.set('Q', ssa.PQ.Q * scale_factor)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.V.bus.v

# Identify min and max bus voltages
min_voltage = round(bus_voltages.min(), 6)
max_voltage = round(bus_voltages.max(), 6)
min_bus = int(ssa.V.bus.idx[bus_voltages.argmin()])
max_bus = int(ssa.V.bus.idx[bus_voltages.argmax()])

# Create a bar chart of bus voltages
plt.figure(figsize=(10, 6))
plt.bar(range(len(bus_voltages)), bus_voltages, color='royalblue')
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.xticks(ticks=range(len(bus_voltages)), labels=range(len(bus_voltages)))
plt.grid(axis='y')

# Save the plot
plot_file = 'scenario_079_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the JSON result
result_json = {
    "scale_factor": float(scale_factor),
    "min_bus": min_bus,
    "min_voltage": float(min_voltage),
    "max_bus": max_bus,
    "max_voltage": float(max_voltage),
    "plot_file": plot_file,
}

# Print the result as specified
print(f"RESULT_JSON={json.dumps(result_json)}")
```

This script successfully implements the changes requested. It adjusts the PV setpoint, scales the PQ loads, runs the power flow analysis, creates a bar chart of the bus voltages, and outputs the specified JSON format.