```python
# required_dependencies: pandas
import andes
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load before setup
pq_load_idx = ssa.PQ.add(bus=10, idx='PQ_VERIFY_041_D', p0=0.016, q0=0.01)

# Re-run power flow
andes.run_power_flow(ssa)

# Retrieve the slack bus voltage and buses below 0.960 p.u.
slack_bus = ssa.Bus.idx[ssa.slack.idx].v[0]
slack_voltage = ssa.Bus.v[ssa.slack.idx].v[0]
threshold = 0.960
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v < threshold].tolist()
selected_count = len(selected_bus_ids)

# Prepare the result
RESULT_JSON = {
    "added_load_idx": pq_load_idx,
    "slack_bus": slack_bus,
    "slack_voltage": round(slack_voltage, 6),
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}

# Print the result in the specified JSON format
print("RESULT_JSON=", RESULT_JSON)
```