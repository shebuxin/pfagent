# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the Kundur full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6
pq_load_idx = ssa.PQ.add(bus=6, p0=0.016, q0=0.01, idx='PQ_VERIFY_069_D')

# Set the slack-bus voltage target to 1.000
ssa.slack_bus.target_voltage.set(1.000)

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Retrieve slack bus voltage
slack_voltage = round(float(ssa.bus.voltage.v[ssa.slack_bus.idx.v[0]]), 6)
slack_setpoint = 1.000

# Retrieve all bus voltage data for plotting
voltages = ssa.bus.voltage.v
bus_ids = ssa.bus.idx.v

# Find the three lowest voltage buses
lowest_voltage_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])[:3]
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_indices]
selected_voltages = [round(float(voltages[i]), 6) for i in lowest_voltage_indices]

# Create line plot of bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o', label='Bus Voltages')
plt.axhline(y=0.950, color='r', linestyle='--', label='0.950 p.u. Threshold')
plt.title('Bus Voltages After Load Addition')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.legend()
plt.grid()
plt.savefig('scenario_069_turn3_line.png')

# Prepare and print the result in the required format
RESULT_JSON = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_069_turn3_line.png'
}
print("RESULT_JSON=", RESULT_JSON)