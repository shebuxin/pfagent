```python
# required_dependencies: pandas, andes
import pandas as pd
import andes

# Load the case
case = 'verify_kundur_078.xlsx'
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
slack_setpoint = 1.020
ssa.Bus.v_set.v[slack_bus_idx] = slack_setpoint

# Add a new PQ load at bus 9
new_load_idx = ssa.PQ.idx_next.v
ssa.PQ.add(idx='PQ_VERIFY_078_B', bus=9, p0=0.018, q0=0.013)

# Rerun power flow
andes.run_power_flow(ssa)

# Get voltage data
voltages = ssa.Bus.v.v

# Identify max and min voltage buses
max_bus = voltages.argmax()
min_bus = voltages.argmin()
max_voltage = round(voltages[max_bus], 6)
min_voltage = round(voltages[min_bus], 6)

# Count total PQ loads present
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON-like structure
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```