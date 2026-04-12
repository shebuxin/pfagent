```python
# required_dependencies: andes
import andes
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_047.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count buses above threshold
threshold = 1.020
selected_bus_ids = ssa.BUS.idx.v[ssa.BUS.vm.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find two lowest-voltage buses
lowest_voltage_indices = ssa.BUS.vm.v.argsort()[:2]
lowest_bus_ids = ssa.BUS.idx.v[lowest_voltage_indices].tolist()
lowest_voltages = [round(ssa.BUS.vm.v[idx], 6) for idx in lowest_voltage_indices]

# Prepare the result JSON object
RESULT_JSON = {
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages,
}

# Print the result
print(f'RESULT_JSON={RESULT_JSON}')
```