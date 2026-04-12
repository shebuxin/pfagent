```python
# required_dependencies: andes
import andes

# Load the Kundur full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6
pq_load_idx = ssa.PQ.add(bus=6, p0=0.016, q0=0.01, idx='PQ_VERIFY_069_D')

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Retrieve slack bus information
slack_bus = ssa.bus.idx.v[ssa.slack_bus.idx.v[0]]
slack_voltage = round(float(ssa.bus.voltage.v[ssa.slack_bus.idx.v[0]]), 6)

# Find all buses below 0.950 p.u.
threshold = 0.950
selected_bus_indices = [i for i, v in enumerate(ssa.bus.voltage.v) if v < threshold]
selected_bus_ids = [int(ssa.bus.idx.v[i]) for i in selected_bus_indices]
selected_count = len(selected_bus_ids)

# Prepare and print the result in the required format
RESULT_JSON = {
    "added_load_idx": pq_load_idx,
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count
}
print("RESULT_JSON=", RESULT_JSON)
```