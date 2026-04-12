Here’s the complete runnable Python script that sets the slack-bus voltage target, reruns the power flow, and saves the line plot of bus voltages.

```python
# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7 if not already added
pq_load_idx = 'PQ_VERIFY_070_D'
bus_7_idx = 7  # Bus 7 index
p0 = 0.018
q0 = 0.012

# Add the PQ load
ssa.PQ.add(buses=bus_7_idx, p0=p0, q0=q0, idx=pq_load_idx)

# Set slack-bus voltage target
voltage_target = 1.010
ssa.Bus.v_target.v[ssa.Bus.slack_flag.v] = voltage_target

# Setup modified case
ssa.setup()

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage
slack_bus_idx = ssa.bus.v[ssa.Bus.slack_flag.v].idx.v[0]
slack_voltage = round(ssa.Bus.v[slack_bus_idx], 6)

# Find the lowest-voltage buses
voltages = ssa.Bus.v
lowest_voltage_indices = voltages.argsort()[:4]
selected_bus_ids = ssa.Bus.idx.v[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in lowest_voltage_indices]

# Save line plot of bus voltages
plt.figure()
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_070_turn3_line.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "slack_setpoint": voltage_target,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_070_turn3_line.png'
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```

This script maintains the previously added load, sets the slack-bus voltage target, reruns the power flow, and generates a line plot of the bus voltages, along with the desired JSON output structure.