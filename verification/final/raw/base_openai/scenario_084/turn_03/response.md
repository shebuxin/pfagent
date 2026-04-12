```python
# required_dependencies: andes,pandas,matplotlib
import andes
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_084.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

# Add a new PQ load at bus 7 (as before)
new_load_idx = 'PQ_VERIFY_084_D'
ssa.PQ.add(idx=new_load_idx, bus=7, p0=0.018, q0=0.012)

# Set the slack-bus voltage target
slack_setpoint = 1.010
ssa.bus.vmag.set(ssa.bus.slack, slack_setpoint)

# Setup the model
andes.setup(ssa)

# Run power flow
andes.run()

# Retrieve slack bus information
slack_bus = ssa.bus.idx[ssa.bus.slack].tolist()[0]
slack_voltage = ssa.bus.vmag.v[ssa.bus.slack]

# Find every bus voltage
bus_voltages = pd.DataFrame({
    'bus_id': ssa.bus.idx.v,
    'voltage': ssa.bus.vmag.v
})

# Find the 4 lowest-voltage buses
lowest_voltages = bus_voltages.nsmallest(4, 'voltage').sort_values(by='voltage')

selected_bus_ids = list(lowest_voltages['bus_id'])
selected_voltages = [round(v, 6) for v in list(lowest_voltages['voltage'])]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_voltages['bus_id'], bus_voltages['voltage'], marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_084_turn3_line.png')
plt.close()

# Create the result JSON
result_json = {
    'slack_setpoint': round(slack_setpoint, 6),
    'slack_voltage': round(slack_voltage, 6),
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages,
    'plot_file': 'scenario_084_turn3_line.png'
}

# Print the result
print(f'RESULT_JSON={result_json}')
```