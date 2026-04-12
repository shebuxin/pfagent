# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the built-in PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor

# Add new PQ load before setup at bus 1
new_load_idx = ssa.PQ.idx.v.max() + 1
ssa.PQ.append(
    idx=new_load_idx,
    bus=1,
    p0=0.012,
    q0=0.006
)

# Run power flow
andes.run_power_flow(ssa)

# Extract bus voltage data
voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Determine max and min voltage buses
max_voltage = voltages.max()
min_voltage = voltages.min()
max_bus = int(bus_ids[voltages.argmax()])
min_bus = int(bus_ids[voltages.argmin()])

# Save line plot of bus voltages
plt.figure(figsize=(10, 5))
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (pu)')
plt.grid()
plt.savefig('scenario_085_turn3_voltage.png')
plt.close()

# Create the result JSON object
RESULT_JSON = {
    'added_load_idx': int(new_load_idx),
    'max_bus': max_bus,
    'max_voltage': round(max_voltage, 6),
    'min_bus': min_bus,
    'min_voltage': round(min_voltage, 6),
    'plot_file': 'scenario_085_turn3_voltage.png'
}

print(f"RESULT_JSON={RESULT_JSON}")