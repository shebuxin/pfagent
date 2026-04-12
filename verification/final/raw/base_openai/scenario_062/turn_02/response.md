```python
# required_dependencies: pandas, numpy
import andes
import pandas as pd
import numpy as np

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.000
slack_bus_index = ssa.Bus.idx.v[0]  # Assuming the slack bus is the first one
ssa.Bus.voltage_target.set(slack_bus_index, 1.000)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the slack bus voltage
slack_voltage = round(ssa.Bus.voltage.v[slack_bus_index], 6)

# Count how many buses fall below 0.950 p.u.
count_below_threshold = np.sum(ssa.Bus.voltage.v < 0.950)

# Prepare RESULT_JSON
RESULT_JSON = {
    "slack_bus": slack_bus_index,
    "slack_setpoint": 1.000,
    "slack_voltage": slack_voltage,
    "selected_count": count_below_threshold
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")
```