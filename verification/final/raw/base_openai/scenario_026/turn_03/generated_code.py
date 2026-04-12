# required_dependencies: andes, matplotlib
import andes
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_026.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add the PQ load if not already added (from the previous step)
bus_idx = np.where(ssa.Bus.idx.v == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 not found.")

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.slack.v
ssa.Bus.Vtarget.set(slack_bus_idx, 1.020)

# Rerun power flow analysis
andes.setup(ssa)
andes.run_power_flow(ssa)

# Extract slack voltage
slack_voltage = round(float(ssa.Bus.V.v[slack_bus_idx - 1]), 6)  # -1 for zero indexing

# Extract bus voltage data and identify the two lowest voltage buses
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v
voltage_data = list(zip(bus_ids, voltages))
lowest_voltage_buses = sorted(voltage_data, key=lambda x: x[1])[:2]

# Prepare result data for JSON
selected_bus_ids = [int(bus[0]) for bus in lowest_voltage_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_voltage_buses]

# Create the plot of bus voltages
plt.figure(figsize=(10, 6))
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.axhline(y=1.020, color='r', linestyle='--', label='Slack Target Voltage (1.020 p.u.)')
plt.legend()
plt.savefig('scenario_026_turn3_line.png')
plt.close()

# Create RESULT_JSON
RESULT_JSON = {
    'slack_setpoint': 1.020,
    'slack_voltage': slack_voltage,
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages,
    'plot_file': 'scenario_026_turn3_line.png'
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)