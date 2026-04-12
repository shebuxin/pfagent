# required_dependencies: andes, matplotlib
import andes
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_083.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6 (if not already added)
bus_idx = np.where(ssa.Bus.idx.v == 6)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 6 not found in the system.")
sssa.PQ.add(idx='PQ_VERIFY_083_D', bus=6, p0=0.016, q0=0.01)

# Set slack-bus voltage target
slack_idx = ssa.Slack.idx.v[0]
ssa.Slack.Vtarget.v[slack_idx] = 1.000

# Run power flow
andes.run(ssa)

# Obtain the slack bus voltage
slack_voltage = round(ssa.Bus.V.v[slack_idx], 6)

# Identify buses and their voltages for plotting
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v

# Find the 3 lowest-voltage buses
lowest_indices = np.argsort(voltages)[:3]
selected_bus_ids = bus_ids[lowest_indices].tolist()
selected_voltages = np.round(voltages[lowest_indices].tolist(), 6).tolist()

# Plot bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_083_turn3_line.png')
plt.close()

# Prepare result in JSON format
result_json = {
    "slack_setpoint": 1.000,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_083_turn3_line.png'
}

print(f"RESULT_JSON={json.dumps(result_json)}")