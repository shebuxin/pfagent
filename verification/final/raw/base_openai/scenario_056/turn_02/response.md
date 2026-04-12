```python
# required_dependencies: andes
import andes
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_056.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load at bus 15
bus_number = 15
load_idx = 'PQ_VERIFY_056_D'
p0 = 0.018
q0 = 0.012

# Find the correct bus index for bus number 15
bus_idx = np.where(ssa.Bus.idx.v == bus_number)[0]
if bus_idx.size == 0:
    raise ValueError(f"Bus number {bus_number} not found.")
    
# Add the PQ load
ssa.PQ.add(bus=bus_number, idx=load_idx, p0=p0, q0=q0)

# Setup the case again and run power flow analysis
ssa.setup()
andes.run_power_flow(ssa)

# Retrieve the slack bus info
slack_bus = ssa.Bus.slack.v
slack_voltage = round(ssa.Bus.voltage.v[slack_bus], 6)

# Find all buses below 0.970 p.u.
threshold = 0.970
lower_buses = [(ssa.Bus.idx.v[i], round(ssa.Bus.voltage.v[i], 6)) 
               for i in range(ssa.Bus.idx.v.size) if ssa.Bus.voltage.v[i] < threshold]

selected_bus_ids = [bus[0] for bus in lower_buses]
selected_count = len(selected_bus_ids)

# Prepare the result
RESULT_JSON = {
    "added_load_idx": load_idx,
    "slack_bus": slack_bus,
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")
```