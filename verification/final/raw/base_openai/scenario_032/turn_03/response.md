```python
# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 20
added_load_idx = "PQ_VERIFY_032_A"
ssa.PQ.add(bus=20, p0=0.019, q0=0.012, id=added_load_idx)

# Rerun power flow
andes.run_power_flow(ssa)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Rerun power flow again
andes.run_power_flow(ssa)

# Extract bus voltages for plotting
voltages = ssa.bus.Vmag.v
bus_ids = ssa.bus.idx.v

# Finding max and min voltage buses
max_bus_index = voltages.index(max(voltages))
min_bus_index = voltages.index(min(voltages))
max_bus = int(bus_ids[max_bus_index])
max_voltage = round(float(voltages[max_bus_index]), 6)
min_bus = int(bus_ids[min_bus_index])
min_voltage = round(float(voltages[min_bus_index]), 6)

# Save line plot of bus voltage magnitude
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plot_file = 'scenario_032_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare Result JSON
RESULT_JSON = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

print("RESULT_JSON=", RESULT_JSON)
```