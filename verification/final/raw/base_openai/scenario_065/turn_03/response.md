```python
# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the built-in Kundur case
case = andes.get_case("kundur/kundur_aw.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 0.990
ssa.PV.set(0, target_voltage=0.990)

# Scale every PQ load by 1.030
ssa.PQ.set_scaler(1.030)

# Rerun power flow
andes.run_powerflow(ssa)

# Get bus voltage information
voltages = ssa.Bus.v
min_voltage = round(float(voltages.min()), 6)
max_voltage = round(float(voltages.max()), 6)
min_bus = int(ssa.Bus.idx.v[voltages.argmin()])
max_bus = int(ssa.Bus.idx.v[voltages.argmax()])

# Save a bar chart of the bus voltages
plt.figure(figsize=(10, 6))
plt.bar(ssa.Bus.idx.v, voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Bus Voltage (p.u.)')
plt.title('Bus Voltages After Scaling PQ Loads')
plt.ylim(0.8, 1.2)
plt.grid()
plot_file = 'scenario_065_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result
result_json = {
    "scale_factor": 1.030,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": plot_file,
}

# Print result
print(f"RESULT_JSON={result_json}")
```