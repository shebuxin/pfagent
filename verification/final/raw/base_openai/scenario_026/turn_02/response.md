```python
# required_dependencies: andes
import andes
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_026.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
bus_idx = np.where(ssa.Bus.idx.v == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 not found.")

ssa.PQ.add(idx="PQ_VERIFY_026_D", bus=4, p0=0.014, q0=0.008)

# Set up and run power flow analysis
andes.setup(ssa)
andes.run_power_flow(ssa)

# Extract slack bus data
slack_bus_idx = ssa.Bus.slack.v
slack_voltage = round(float(ssa.Bus.V.v[slack_bus_idx - 1]), 6)  # -1 for zero indexing

# Identify buses with voltage below 1.000 p.u.
threshold = 1.000
selected_bus_ids = []
for idx, voltage in enumerate(ssa.Bus.V.v):
    if voltage < threshold:
        selected_bus_ids.append(int(ssa.Bus.idx.v[idx]))

selected_count = len(selected_bus_ids)

# Create RESULT_JSON
RESULT_JSON = {
    'added_load_idx': "PQ_VERIFY_026_D",
    'slack_bus': int(slack_bus_idx),
    'slack_voltage': slack_voltage,
    'threshold': threshold,
    'selected_bus_ids': selected_bus_ids,
    'selected_count': selected_count
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```