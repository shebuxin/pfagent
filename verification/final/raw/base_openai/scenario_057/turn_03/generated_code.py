# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
load_idx = ssa.PQ.add("PQ_VERIFY_057_A", bus=4, p0=0.01, q0=0.006)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
for idx in range(ssa.PQ.idx.v.size):
    ssa.PQ.p0.v[idx] *= scale_factor
    ssa.PQ.q0.v[idx] *= scale_factor

# Run the power flow again
andes.run_power_flow(ssa)

# Gather voltage data
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find max and min voltage buses
max_voltage_data = max(zip(bus_ids, bus_voltages), key=lambda x: x[1])
min_voltage_data = min(zip(bus_ids, bus_voltages), key=lambda x: x[1])

# Prepare values for the result
max_bus = max_voltage_data[0]
max_voltage = round(max_voltage_data[1], 6)
min_bus = min_voltage_data[0]
min_voltage = round(min_voltage_data[1], 6)

# Create a line plot of bus voltage magnitudes
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_057_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")