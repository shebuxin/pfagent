# required_dependencies: andes, matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
new_load = ssa.PQ.add(bus=5, idx='PQ_VERIFY_013_D', p0=0.016, q0=0.01)

# Set the slack-bus voltage target
slack_setpoint = 1.025
ssa.Bus.V.set(slack_setpoint, idx=ssa.Bus.slack_idx.v[0])

# Run power flow analysis
andes.run_power_flow(ssa)

# Get the slack bus and its voltage
slack_bus = ssa.Bus.slack_idx.v[0]
slack_voltage = ssa.Bus.V.v[slack_bus]

# Identify the buses and their voltages
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v

# Find the indices of the lowest-voltage buses
lowest_voltage_indices = np.argsort(voltages)[:3]
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[i], 6) for i in lowest_voltage_indices]

# Plot bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_013_turn3_line.png')

# Construct the result JSON
RESULT_JSON = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_013_turn3_line.png'
}
print("RESULT_JSON=", RESULT_JSON)