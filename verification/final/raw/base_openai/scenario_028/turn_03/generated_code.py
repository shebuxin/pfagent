# required_dependencies: pandas, matplotlib
import os
import andes
import pandas as pd
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_028.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add the PQ load at bus 9 (already done in the previous step)
new_load_idx = 'PQ_VERIFY_028_D'
ssa.PQ.add(bus=9, idx=new_load_idx, p0=0.018, q0=0.012)

# Set the slack-bus voltage target to 1.035
slack_bus = ssa.Bustype.idx[0]  # Assuming the first bus is the slack bus
ssa.Bus.v[slack_bus] = 1.035

# Run power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.v[slack_bus], 6)

# Get voltages and bus IDs for the lowest-voltage buses
voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx
lowest_voltages_indices = voltages.argsort()[:4]
selected_bus_ids = bus_ids[lowest_voltages_indices].tolist()
selected_voltages = [round(voltages[i], 6) for i in lowest_voltages_indices]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_028_turn3_line.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "slack_setpoint": 1.035,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_028_turn3_line.png'
}

print(f"RESULT_JSON={RESULT_JSON}")