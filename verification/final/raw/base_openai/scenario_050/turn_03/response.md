```python
# required_dependencies: numpy, pandas, json
import os
import numpy as np
import pandas as pd
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus_idx = ssa.Bus.idx.v[np.where(ssa.Bus.type.v == 'slack')[0][0]]  # Assuming only one slack bus
slack_setpoint = 1.035
ssa.VM.set(slack_bus_idx, slack_setpoint)

# Add new PQ load at bus 20
pq_load_idx = ssa.PQ.add(p0=0.018, q0=0.013, bus=20, idx='PQ_VERIFY_050_B')

# Rerun power flow
andes.run(ssa)

# Get maximum and minimum voltage buses
max_bus_idx = np.argmax(ssa.Bus.v.v)
min_bus_idx = np.argmin(ssa.Bus.v.v)
max_voltage = round(ssa.Bus.v.v[max_bus_idx], 6)
min_voltage = round(ssa.Bus.v.v[min_bus_idx], 6)

# Total count of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Constructing the result JSON
result_json = {
    "added_load_idx": pq_load_idx,
    "max_bus": max_bus_idx,
    "max_voltage": max_voltage,
    "min_bus": min_bus_idx,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count,
}

# Printing the result
print("RESULT_JSON=" + json.dumps(result_json))
```