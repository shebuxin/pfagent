# required_dependencies: andes, matplotlib
import andes
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
new_load_idx = ssa.PQ.add('PQ_VERIFY_001_A', p0=0.01, q0=0.006, bus=4)

# Scale every PQ load by a factor of 1.030
scale_factor = 1.030
ssa.PQ.p.v *= scale_factor
ssa.PQ.q.v *= scale_factor

# Run power flow
andes.run_power_flow(ssa)

# Get all bus voltages
voltages = ssa.bus.v

# Identify max and min bus voltages
max_bus = int(np.argmax(voltages))
max_voltage = round(float(voltages[max_bus]), 6)
min_bus = int(np.argmin(voltages))
min_voltage = round(float(voltages[min_bus]), 6)

# Save line plot of bus voltage magnitude
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(voltages)), voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_001_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
result_json = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")