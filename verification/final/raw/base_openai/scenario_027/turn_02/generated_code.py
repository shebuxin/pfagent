# required_dependencies: andes,numpy
import andes
import numpy as np

# Load the uploaded case
case = "verify_ieee14_027.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 5
new_load_idx = 'PQ_VERIFY_027_D'
p0 = 0.016
q0 = 0.01
ssa.PQ.add(bus=5, idx=new_load_idx, p0=p0, q0=q0)

# Run power flow again
andes.run_power_flow(ssa)

# Retrieve slack bus information
slack_bus = ssa.Bus.slack.v
slack_voltage = ssa.Bus.v[slack_bus]

# Find all buses with voltage below 1.005 p.u.
threshold = 1.005
selected_bus_indices = np.where(ssa.Bus.v < threshold)[0]
selected_bus_ids = ssa.Bus.idx.v[selected_bus_indices].tolist()
selected_count = len(selected_bus_ids)

# Create RESULT_JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "slack_bus": slack_bus,
    "slack_voltage": round(slack_voltage, 6),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the results
print(f'RESULT_JSON={RESULT_JSON}')