# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
added_load_idx = 'PQ_VERIFY_002_A'
p0 = 0.013
q0 = 0.008
bus_idx = ssa.Bus.idx.v.tolist().index(5)  # Find the index of bus 5
ssa.PQ.add(added_load_idx, bus=bus_idx, p0=p0, q0=q0)

# Scale every PQ load by a factor of 1.040
scale_factor = 1.040
for pq in ssa.PQ.idx.v:
    ssa.PQ.set(pq, p0=ssa.PQ.p0.v[pq] * scale_factor, q0=ssa.PQ.q0.v[pq] * scale_factor)

# Run power flow again
andes.run_power_flow(ssa)

# Collect bus voltages
voltages = ssa.Bus.v

# Find max and min voltage buses
max_bus_idx = voltages.index(max(voltages))
min_bus_idx = voltages.index(min(voltages))
max_bus = ssa.Bus.idx.v[max_bus_idx]
min_bus = ssa.Bus.idx.v[min_bus_idx]
max_voltage = round(voltages[max_bus_idx], 6)
min_voltage = round(voltages[min_bus_idx], 6)

# Plot bus voltage magnitudes
plt.figure(figsize=(10, 6))
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_002_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")