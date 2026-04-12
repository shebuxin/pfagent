```python
# required_dependencies: andes,json,matplotlib
import andes
import json
import matplotlib.pyplot as plt

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add the new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_060_A'
p0 = 0.019
q0 = 0.012
bus_9_idx = np.where(ssa.PQ.bus.v == 9)[0][0]
ssa.PQ.add(new_load_idx, bus_9_idx, p0, q0)

# Setup is required before running power flow
andes.setup(ssa)

# Scale every PQ load by a factor of 1.060
scale_factor = 1.060
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Run power flow
andes.run_power_flow(ssa)

# Get bus voltages
bus_voltages = ssa.PQ.v.v
bus_ids = ssa.PQ.bus.v

# Identify max and min voltage buses
max_voltage_idx = np.argmax(bus_voltages)
min_voltage_idx = np.argmin(bus_voltages)

max_bus = int(bus_ids[max_voltage_idx])
max_voltage = round(bus_voltages[max_voltage_idx], 6)
min_bus = int(bus_ids[min_voltage_idx])
min_voltage = round(bus_voltages[min_voltage_idx], 6)

# Save a line plot of bus voltage magnitude
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_060_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
result_json = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))
```