```python
# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
result = andes.run_power_flow(ssa)

# Find max and min voltage buses
max_voltage_idx = result.V.idx.v[np.argmax(result.V.mag.v)]
min_voltage_idx = result.V.idx.v[np.argmin(result.V.mag.v)]
max_voltage = round(result.V.mag.v[max_voltage_idx], 6)
min_voltage = round(result.V.mag.v[min_voltage_idx], 6)

# Prepare result JSON
result_json = {
    "max_bus": int(max_voltage_idx),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_voltage_idx),
    "min_voltage": float(min_voltage)
}

# Print result
print("RESULT_JSON=" + json.dumps(result_json))
```