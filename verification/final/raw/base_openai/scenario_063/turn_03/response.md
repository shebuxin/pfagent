```python
# required_dependencies: pandas
import pandas as pd
import andes

# Load the built-in Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Slack.idx.v[0]  # Assuming there's only one slack bus
slack_setpoint = 1.010
ssa.Slack.set_voltage_target(slack_bus_idx, slack_setpoint)

# Add a new PQ load at bus 7
pq_load_idx = 'PQ_VERIFY_063_B'
p0, q0 = 0.016, 0.011
ssa.PQ.add(bus=7, idx=pq_load_idx, p0=p0, q0=q0)

# Rerun power flow
andes.run_power_flow(ssa)

# Retrieve maximum and minimum voltage buses
voltages = ssa.Bus.voltage.v
max_bus = ssa.Bus.idx.v[voltages.argmax()]
min_bus = ssa.Bus.idx.v[voltages.argmin()]
max_voltage = round(voltages.max(), 6)
min_voltage = round(voltages.min(), 6)

# Count total PQ loads present
total_pq_count = len(ssa.PQ.idx.v)

# Constructing the final JSON-like output
RESULT_JSON = {
    "added_load_idx": pq_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print(f"RESULT_JSON={RESULT_JSON}")
```