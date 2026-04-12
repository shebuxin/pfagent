# required_dependencies: pandas, matplotlib
import andes
import pandas as pd
import matplotlib.pyplot as plt

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
bus_idx = np.where(ssa.Bus.idx.v == 4)[0][0]
load_idx = ssa.PQ.add(idx='PQ_VERIFY_068_D', bus=bus_idx, p0=0.014, q0=0.008)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v[0]
ssa.Slack.v0.set(v0=0.990)

# Rerun the power flow
andes.run("powerflow", ssa)

# Get the slack bus voltage
slack_voltage = ssa.Bus.vmag.v[slack_bus_idx]

# Find the bus voltages
bus_voltages = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.vmag.v
})

# Prepare to save the line plot
plt.figure(figsize=(10, 5))
plt.plot(bus_voltages['bus_id'], bus_voltages['voltage'], marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.axhline(y=0.990, color='r', linestyle='--', label='Slack Setpoint')
plt.legend()
plt.grid()
plot_file = 'scenario_068_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Get the two lowest voltage buses
lowest_buses = bus_voltages.nsmallest(2, 'voltage').sort_values('voltage')

# Prepare the result
RESULT_JSON = {
    "slack_setpoint": 0.990,
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": lowest_buses['bus_id'].tolist(),
    "selected_voltages": [round(v, 6) for v in lowest_buses['voltage'].tolist()],
    "plot_file": plot_file
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")