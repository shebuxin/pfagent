```python
# required_dependencies: andes,json
import andes
import json
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
bus_number = 15
load_idx = 'PQ_VERIFY_031_A'
p0 = 0.016
q0 = 0.01
ssa.PQ.add(bus=bus_number, idx=load_idx, p0=p0, q0=q0)

# Setup case after adding the load
andes.setup(ssa)

# Run power flow
andes.run_power_flow(ssa)

# Identify threshold and find buses below it
threshold = 0.970
bus_voltages = ssa.Bus.Vmag.v
buses_below_threshold = np.where(bus_voltages < threshold)[0]
selected_bus_ids = ssa.Bus.idx.v[buses_below_threshold].astype(int).tolist()

# Find the minimum voltage bus
min_bus_idx = np.argmin(bus_voltages)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Construct the result JSON
result_json = {
    "added_load_idx": load_idx,
    "added_load_bus": bus_number,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage,
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))
```