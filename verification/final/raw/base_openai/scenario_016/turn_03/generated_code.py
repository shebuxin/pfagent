# required_dependencies: pandas,json,andes,matplotlib
import pandas as pd
import json
import andes
import matplotlib.pyplot as plt

case = "verify_ieee14_016.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5 (carried over from previous step)
new_load_idx = 'PQ_VERIFY_016_A'
p0 = 0.013
q0 = 0.008
bus_number = 5

bus_idx = np.where(ssa.Bus.idx.v == bus_number)[0]
if bus_idx.size == 0:
    raise ValueError(f"Bus {bus_number} not found.")
bus_idx = bus_idx[0]

# Set the new load
ssa.PQ.add(new_load_idx, p0, q0, bus_idx)

# Scale all PQ loads by a factor of 1.040
scale_factor = 1.040
ssa.PQ.set('p0', ssa.PQ.p0.v * scale_factor)
ssa.PQ.set('q0', ssa.PQ.q0.v * scale_factor)

# Run power flow after scaling
andes.run_power_flow()

# Get bus voltages
voltages = ssa.Bus.V.v

# Find max and min voltage buses
max_voltage_idx = np.argmax(voltages)
min_voltage_idx = np.argmin(voltages)

max_bus = ssa.Bus.idx.v[max_voltage_idx]
max_voltage = round(voltages[max_voltage_idx], 6)
min_bus = ssa.Bus.idx.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Generate a line plot for bus voltage magnitudes
plt.figure()
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_016_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "plot_file": plot_file
}

print(f"RESULT_JSON={json.dumps(RESULT_JSON)}")