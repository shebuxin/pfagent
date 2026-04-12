# required_dependencies: pandas, matplotlib
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load before setup
pq_load_idx = ssa.PQ.add(bus=10, idx='PQ_VERIFY_041_D', p0=0.016, q0=0.01)

# Set the slack-bus voltage target
slack_setpoint = 1.020
ssa.slack.set_vtarget(slack_setpoint)

# Re-run power flow
andes.run_power_flow(ssa)

# Retrieve the slack bus voltage and buses below 0.960 p.u.
slack_voltage = ssa.Bus.v[ssa.slack.idx].v[0]
threshold = 0.960
selected_bus_ids = ssa.Bus.idx.v[np.argsort(ssa.Bus.v)[:3]].tolist()
selected_voltages = np.round(ssa.Bus.v[selected_bus_ids].tolist(), 6).tolist()

# Plotting bus voltages
plt.figure()
plt.plot(ssa.Bus.idx.v, ssa.Bus.v, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_041_turn3_line.png')

# Prepare the result
RESULT_JSON = {
    "slack_setpoint": round(slack_setpoint, 6),
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_041_turn3_line.png'
}

# Print the result in the specified JSON format
print("RESULT_JSON=", RESULT_JSON)